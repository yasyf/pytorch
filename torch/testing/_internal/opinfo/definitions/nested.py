# mypy: ignore-errors

from copy import copy
from dataclasses import dataclass
from enum import Enum
from functools import partial
from itertools import product
from typing import List, Set

import torch
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.opinfo.core import (
    BinaryUfuncInfo,
    ReductionOpInfo,
    SampleInput,
    UnaryUfuncInfo,
)
from torch.utils._pytree import tree_map


# Specifies each type of dim relevant to an NJT's shape
class DimType(Enum):
    BATCH_DIM = 0
    RAGGED_DIM = 1
    # AKA non-batch, non-ragged
    NORMAL_DIM = 2

    def __str__(self):
        return self.name.lower()


class Contiguity(Enum):
    CONTIG = 0
    NONCONTIG_TRANSPOSED = (1,)
    NONCONTIG_HOLES = (2,)

    def __str__(self):
        return self.name.lower()


@dataclass
class ExtraOpData:
    """
    Contains info on top of the typical OpInfo data that is useful for NJT test generation.

    The process that converts the standard op_db -> an NJT-compatible op_db will attach this
    data onto each associated OpInfo entry.
    """

    # Indicates whether the associated op is a view op
    is_view: bool = False

    # Specifies the names of any dim-related args that the op takes in. This is useful
    # for NJT tests because there is often asymmetry across the supported set of dims for
    # an op; it may make sense to operate over the batch dim but not the ragged dim, for
    # example. The length of this list should match the number of relevant overloads.
    # Each list item of the outer list should specify dim argnames. Ellipses should be used
    # to indicate multi-dim support for a given overload.
    #
    # For example, squeeze() has both a dim and multi-dim overload, where the argname for
    # each is simply "dim". Its entry should be: [["dim"], ["dim", "..."]].
    #
    # If no overload of the op accepts dim-related args, this should be None.
    dim_args: List[List[str]] = None

    # The set of dim types this op is expected to -ideally- support (i.e. those it makes
    # conceptual sense to support, disregarding any current coverage gaps).
    # If the op doesn't operate dim-wise (i.e. it has no dim-related args), then this should
    # be set to None. All dim types are assumed to be supported by default for dim-wise ops.
    dim_support: Set[DimType] = None

    # The set of contiguity types this op is expected to -ideally- support, disregarding any
    # current coverage gaps. All contiguity types are assumed to be supported by default for
    # all ops.
    contiguity_support: Set[Contiguity] = None

    def __post_init__(self):
        if self.dim_args is not None and self.dim_support is None:
            # assume the op should ideally operate over all dims by default
            self.dim_support = {
                DimType.BATCH_DIM,
                DimType.RAGGED_DIM,
                DimType.NORMAL_DIM,
            }
        if self.contiguity_support is None:
            self.contiguity_support = {
                Contiguity.CONTIG,
                Contiguity.NONCONTIG_TRANSPOSED,
                Contiguity.NONCONTIG_HOLES,
            }


# random integer used for sizes
def _rnd():
    return torch.randint(3, 8, ()).item()


def _raggedness_matches(nt1, nt2):
    return (
        nt1.is_nested
        and nt2.is_nested
        and nt1._ragged_idx == nt2._ragged_idx
        and nt1.shape[nt1._ragged_idx] == nt2.shape[nt2._ragged_idx]
    )


# Reparametrizer that automatically parametrizes over the appropriate dim_type
# for each op that supports batch-wise operation and contiguity type for all ops.
# This is intended to be used with torch.testing._internal.common_utils.reparametrize.
def include_dim_type_and_contiguity(test_name, param_kwargs):
    op = param_kwargs["op"]
    extra_op_data = op._extra_op_data

    dim_types = extra_op_data.dim_support or [None]
    for contiguity, dim_type in product(extra_op_data.contiguity_support, dim_types):
        new_param_kwargs = dict(param_kwargs)
        new_param_kwargs["dim_type"] = dim_type
        dim_type_suffix = "" if dim_type is None else f"_{str(dim_type)}"
        new_param_kwargs["contiguity"] = contiguity
        contiguity_suffix = f"_{str(contiguity)}"
        new_test_name = f"{test_name}{dim_type_suffix}{contiguity_suffix}"
        yield (new_test_name, new_param_kwargs)


# Generates a random NT.
# dims should be something like [5, None, 10], with None indicating that a
# random ragged structure should be used
def random_nt_from_dims(
    dims, device=None, dtype=None, layout=torch.strided, requires_grad=False
):
    sizes = [[d if d is not None else _rnd() for d in dims[1:]] for d in range(dims[0])]
    return torch.nested.nested_tensor(
        [torch.randn(*size) for size in sizes],
        device=device,
        dtype=dtype,
        layout=layout,
        requires_grad=requires_grad,
    )


# Helper function for generating a comprehensive set of NJT sample inputs.
def _sample_njts(
    device,
    dtype,
    requires_grad=False,
    dims=None,
    contiguity=Contiguity.CONTIG,
):
    if dims is None:
        dims = [2, 3, 4]
    if not isinstance(dims, (list, tuple)):
        dims = [dims]

    # contiguous NJTs
    for dim in dims:
        # with min / max seqlen cached
        shape = (_rnd(), None, *[_rnd() for _ in range(dim - 2)])
        nt = random_nt_from_dims(
            shape,
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
            layout=torch.jagged,
        )
        if contiguity == Contiguity.CONTIG:
            yield nt

        # without min / max seqlen cached
        values = nt.values().clone().detach()
        offsets = nt.offsets().clone().detach()
        if contiguity == Contiguity.CONTIG:
            yield torch.nested.nested_tensor_from_jagged(values, offsets)

        # non-contiguous transposed NJT (not possible for 2D)
        if contiguity == Contiguity.NONCONTIG_TRANSPOSED and dim > 2:
            yield nt.transpose(-2, -1)

        # non-contiguous with holes NJT
        if contiguity == Contiguity.NONCONTIG_HOLES:
            values = nt.values().clone().detach()
            offsets = nt.offsets().clone().detach()
            # subtract 1 to cause holes
            lengths = (offsets.diff() - 1).clone().detach()
            yield torch.nested.nested_tensor_from_jagged(
                values=values,
                offsets=offsets,
                lengths=lengths,
            )


# Computes an unbind-based reference for a given OpInfo on a given SampleInput.
# This reference unbinds the input NJT and invokes the op on each of the components,
# optionally wrapping the result in an NJT.
def unbind_reference(op, sample, wrap_output_as_njt=True):
    # first NJT in the arglist determines expected ragged structure
    nt_inp = (
        sample.input
        if sample.input.is_nested
        # TODO: look in kwargs too?
        else next(a for a in sample.args if a.is_nested)
    )

    out_ref_components = []
    for i in range(nt_inp.shape[0]):

        def _slice_input(t, i=i, inp=nt_inp):
            # any NJT with the same ragged structure as the input should
            # be sliced to pass to the reference
            if isinstance(t, torch.Tensor) and _raggedness_matches(t, inp):
                return t[i]
            # allow the SampleInput to tell us how to slice it for ref calculation
            elif isinstance(t, torch.Tensor) and hasattr(t, "_batch_dim"):
                bdim = t._batch_dim  # type: ignore[attr]
                if t.shape[bdim] == 1:
                    return t[0]
                else:
                    return t.select(bdim, i)
            else:
                return t

        inp = _slice_input(sample.input)
        args = tree_map(_slice_input, sample.args)
        kwargs = tree_map(_slice_input, sample.kwargs)

        # Handle indices in index_put
        if "index_put" in op.full_name and "indices" in kwargs:
            if len(kwargs["indices"]) > 1:
                # If after unrolling we still have indices left, use them
                kwargs["indices"] = [t[i] for t in kwargs["indices"][1:]]
            else:
                # If no indices are left, create them so they match the NJT implementation
                sequence_put = kwargs["indices"][0].tolist()
                if i in sequence_put:
                    kwargs["indices"] = [
                        torch.tensor(
                            list(range(inp.shape[0])),
                            dtype=torch.int32,
                            device=kwargs["indices"][0].device,
                        )
                    ]
                else:
                    kwargs["indices"] = [
                        torch.tensor(
                            [], dtype=torch.int32, device=kwargs["indices"][0].device
                        )
                    ]

        from torch.nested._internal.ops import _outer_to_inner_dim

        # Need to adjust dim to apply on NJT component
        if "dim" in kwargs:
            kwargs["dim"] = _outer_to_inner_dim(
                nt_inp.dim(), kwargs["dim"], canonicalize=True
            )

        # TODO: handle this
        assert "dims" not in kwargs
        out_ref_component = op.op(inp, *args, **kwargs)
        out_ref_components.append(out_ref_component)

    if wrap_output_as_njt:
        # handle list / tuple of outputs
        if len(out_ref_components) > 0 and isinstance(
            out_ref_components[0], (list, tuple)
        ):
            num_returns = len(out_ref_components[0])
            # ensure we get the same number of returns for each invocation
            assert all(len(o) == num_returns for o in out_ref_components)
            # construct NJTs from same index returns from each invocation
            njt_returns = []
            for r in range(num_returns):
                njt_returns.append(
                    torch.nested.as_nested_tensor(
                        [o[r] for o in out_ref_components], layout=torch.jagged
                    )
                )
            return type(out_ref_components[0])(njt_returns)
        return torch.nested.as_nested_tensor(out_ref_components, layout=torch.jagged)

    return out_ref_components


# Computes the reference value for a reduction op.
def reduction_reference(op, sample):
    assert sample.input.is_nested
    dim = sample.kwargs.get("dim", None)
    keepdim = sample.kwargs.get("keepdim", False)
    assert dim != 0, "reductions over the batch dim are not supported"
    assert "dims" not in sample.kwargs
    assert sample.input._ragged_idx == 1

    if isinstance(dim, (tuple, list)):
        reduce_on_ragged = sample.input._ragged_idx in dim
        reduce_on_batch = 0 in dim
    else:
        reduce_on_ragged = sample.input._ragged_idx == dim
        reduce_on_batch = dim == 0

    if dim is None:
        # calculate reference value by running reduction on values buffer
        return op.op(sample.input.values(), *sample.args, **sample.kwargs)

    if reduce_on_ragged and reduce_on_batch:
        # run reference directly on buffer with dims converted to inner space
        from torch.nested._internal.ops import _outer_to_inner_dim

        ref_kwargs = dict(sample.kwargs)
        ref_kwargs["dim"] = _outer_to_inner_dim(
            sample.input.dim(), dim, canonicalize=True
        )
        out = op.op(sample.input.values(), *sample.args, **ref_kwargs)
        if keepdim:
            if isinstance(out, (tuple, list)):
                # some ops return multiple things; unsqueeze all of them
                out = type(out)(o.unsqueeze(sample.input._ragged_idx) for o in out)
            else:
                out = out.unsqueeze(sample.input._ragged_idx)
        return out

    if reduce_on_ragged and not reduce_on_batch:
        # calculate reference value by running an unbind reference and stacking
        out_ref_components = unbind_reference(op, sample, wrap_output_as_njt=False)
        if len(out_ref_components) > 0 and isinstance(
            out_ref_components[0], (tuple, list)
        ):
            # some ops return multiple things; stack all of them
            num_returns = len(out_ref_components[0])
            # ensure we get the same number of returns for each invocation
            assert all(len(o) == num_returns for o in out_ref_components)
            # stack same index returns from each invocation
            stacked_returns = []
            for r in range(num_returns):
                stacked_returns.append(
                    torch.stack([o[r] for o in out_ref_components], dim=0)
                )
            return type(out_ref_components[0])(stacked_returns)
        return torch.stack(out_ref_components, dim=0)

    # unbind reference works for other reductions
    return unbind_reference(op, sample)


def sample_inputs_elementwise_njt_unary(
    op_info, device, dtype, requires_grad, op_kwargs=None, **kwargs
):
    if not op_kwargs:
        op_kwargs = {}

    for contiguity in [Contiguity.CONTIG, Contiguity.NONCONTIG_HOLES, Contiguity.NONCONTIG_TRANSPOSED]:
        for njt in _sample_njts(
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
            dims=[2, 3, 4],
            contiguity=contiguity,
        ):
            sample_name = f"{njt.dim()}D: {contiguity}"
            yield SampleInput(njt, kwargs=dict(op_kwargs), name=sample_name)


def sample_inputs_elementwise_njt_binary(
    op_info, device, dtype, requires_grad, op_kwargs=None, **kwargs
):
    assert kwargs.pop("dim_type") is None

    if not op_kwargs:
        op_kwargs = {}

    for njt1 in _sample_njts(
        device=device, dtype=dtype, requires_grad=requires_grad, dims=[2, 3, 4]
    ):
        # TODO: account for non-contiguous NJTs here
        njt2 = torch.randn_like(njt1)
        yield SampleInput(njt1, args=(njt2,), kwargs=dict(op_kwargs))

        # broadcasting case: (B, j0, ...) with (B, 1, ...)
        t = torch.randn(
            (njt1.shape[0], 1, *njt1.shape[2:]),
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
        )
        # used for slicing in unbind_reference()
        t._batch_dim = 0
        # (NT, T)
        yield SampleInput(njt1, args=(t,), kwargs=dict(op_kwargs))
        # (T, NT)
        yield SampleInput(t, args=(njt1,), kwargs=dict(op_kwargs))

        # broadcasting case: (B, j0, ...) with (1, 1...)
        t = torch.randn(
            [1 for _ in range(njt1.dim())],
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
        )
        # used for slicing in unbind_reference()
        t._batch_dim = 0
        # (NT, T)
        yield SampleInput(njt1, args=(t,), kwargs=dict(op_kwargs))
        # (T, NT)
        yield SampleInput(t, args=(njt1,), kwargs=dict(op_kwargs))

        # broadcasting case: (B, j0, ...) with (...)
        t = torch.randn(
            njt1.shape[2:], device=device, dtype=dtype, requires_grad=requires_grad
        )
        # (NT, T)
        yield SampleInput(njt1, args=(t,), kwargs=dict(op_kwargs))
        # (T, NT)
        yield SampleInput(t, args=(njt1,), kwargs=dict(op_kwargs))

        # broadcasting case: (B, j0, ...) with scalar
        t = torch.randn((), device=device, dtype=dtype, requires_grad=requires_grad)
        # (NT, T)
        yield SampleInput(njt1, args=(t,), kwargs=dict(op_kwargs))
        # (T, NT)
        yield SampleInput(t, args=(njt1,), kwargs=dict(op_kwargs))

    # mixed broadcasting case: (B, j0, 1) with (B, 1, D)
    B = 4
    D = 16
    njt = random_nt_from_dims(
        (B, None, 1),
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
        layout=torch.jagged,
    )
    t = torch.randn(B, 1, D, device=device, dtype=dtype, requires_grad=requires_grad)
    # used for slicing in unbind_reference()
    t._batch_dim = 0

    # (NT, T)
    yield SampleInput(njt, args=(t,), kwargs=dict(op_kwargs))
    # (T, NT)
    yield SampleInput(t, args=(njt,), kwargs=dict(op_kwargs))


def sample_inputs_njt_reduction(
    op_info,
    device,
    dtype,
    requires_grad,
    supports_dimlist=True,
    op_kwargs=None,
    **kwargs,
):
    if not op_kwargs:
        op_kwargs = {}

    for njt in _sample_njts(
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
        dims=[2, 3, 4],
        contiguity=kwargs.pop("contiguity"),
    ):
        for keepdim in [False, True]:
            # single dim-wise reduction; includes reduction over the ragged dim
            # NB: reduction over the batch dim is not supported!
            # TODO: Cover this in the set of error inputs
            for dim in range(1, njt.dim()):
                yield SampleInput(
                    njt.clone().detach(),
                    kwargs={**op_kwargs, "dim": dim, "keepdim": keepdim},
                )

            if supports_dimlist:
                # reduce on both batch and ragged dims
                yield SampleInput(
                    njt.clone().detach(),
                    kwargs={
                        **op_kwargs,
                        "dim": [0, njt._ragged_idx],
                        "keepdim": keepdim,
                    },
                )

                # reduce on batch, ragged, and other dims
                for other_dim in range(njt._ragged_idx + 1, njt.dim()):
                    yield SampleInput(
                        njt.clone().detach(),
                        kwargs={
                            **op_kwargs,
                            "dim": [0, njt._ragged_idx, other_dim],
                            "keepdim": keepdim,
                        },
                    )

                # reduce on two non-ragged, non-batch dims
                if njt.dim() > 3 and njt._ragged_idx == 1:
                    yield SampleInput(
                        njt.clone().detach(),
                        kwargs={
                            **op_kwargs,
                            "dim": [njt.dim() - 2, njt.dim() - 1],
                            "keepdim": keepdim,
                        },
                    )

                # full reduction by specifying all dims
                yield SampleInput(
                    njt.clone().detach(),
                    kwargs={
                        **op_kwargs,
                        "dim": list(range(njt.dim())),
                        "keepdim": keepdim,
                    },
                )

                # TODO: Reducing on ragged dim and non-batch dim is not supported;
                # cover this in the set of error inputs.

        # full reduction
        yield SampleInput(njt.clone().detach(), kwargs=dict(op_kwargs))


def unsupported_sample_inputs_func(op_name):
    def _f(op_info, device, dtype, requires_grad, op_name=op_name, **kwargs):
        raise RuntimeError(
            f"OpInfo for {op_name} does not support NJT. Support can be added by modifying "
            "torch/testing/_internal/opinfo/definitions/nested.py."
        )

    return _f


def unsupported_reference(op_name):
    def _f(op, sample):
        raise RuntimeError(
            f"OpInfo for {op_name} does not define a ref() function. Support can be added by "
            "modifying torch/testing/_internal/opinfo/definitions/nested.py."
        )

    return _f


# === BEGIN OP-SPECIFIC SAMPLE INPUTS FUNCS ===
def sample_inputs_clone(op_info, device, dtype, requires_grad, **kwargs):
    # non-contiguous NJTs
    for njt in _sample_njts(
        device=device, dtype=dtype, requires_grad=requires_grad, dims=[2, 3, 4]
    ):
        yield SampleInput(njt)

    for memory_format in (torch.contiguous_format, torch.preserve_format):
        # construct a "non-contiguous with holes" NJT
        values = torch.randn(
            10, 5, device=device, dtype=dtype, requires_grad=requires_grad
        )
        offsets = torch.tensor([0, 2, 4, 10], device=device, dtype=torch.int64)
        lengths = torch.tensor([2, 1, 3], device=device, dtype=torch.int64)
        njt = torch.nested.nested_tensor_from_jagged(
            values, offsets=offsets, lengths=lengths
        )

        yield SampleInput(njt, kwargs={"memory_format": memory_format})


def sample_inputs_mvl_gamma(p):
    return partial(sample_inputs_elementwise_njt_unary, op_kwargs={"p": p})


def sample_inputs_polygamma_n(n):
    return partial(sample_inputs_elementwise_njt_unary, op_kwargs={"n": n})


def sample_inputs_special_polygamma_n(n):
    return partial(sample_inputs_elementwise_njt_unary, op_kwargs={"n": n})


def sample_inputs_to(op_info, device, dtype, requires_grad, op_kwargs=None, **kwargs):
    for njt in _sample_njts(
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
        dims=[2, 3, 4],
    ):
        other_dtypes = (
            d for d in (torch.float32, torch.half, torch.double) if d is not dtype
        )
        for other_dtype in other_dtypes:
            sample_name = f"{njt.dim()}D: {dtype} -> {other_dtype}"
            yield SampleInput(
                njt.clone().detach(), kwargs={"dtype": dtype}, name=sample_name
            )

        # only include device transfer for CUDA inputs
        if "cuda" in device:
            other_device = "cpu"
            sample_name = f"{njt.dim()}D: {device} -> {other_device}"
            yield SampleInput(
                njt.clone().detach(), kwargs={"device": other_device}, name=sample_name
            )


def sample_inputs_bmm(op_info, device, dtype, requires_grad, op_kwargs=None, **kwargs):
    for njt_3d in _sample_njts(
        device=device, dtype=dtype, requires_grad=requires_grad, dims=[3]
    ):
        # (B, j1, D) x (B, D, E) => (B, j1, E)
        B, D = njt_3d.shape[0], njt_3d.shape[-1]
        E = D + 2
        other = torch.randn(B, D, E, device=device, dtype=dtype)
        # used for slicing in unbind_reference()
        other._batch_dim = 0
        yield SampleInput(njt_3d.clone().detach(), kwargs={"mat2": other})

        # TODO (need factory functions):
        # (B, D, j1) x (B, j1, E) => (B, D, E)


def reference_bmm(op, sample):
    # unbind reduces a dim and bmm requires 3D, so use matmul as the reference
    matmul_op = copy(op)
    matmul_op.op = torch.matmul
    # change arg name from mat2 -> other
    modified_sample = copy(sample)
    other = modified_sample.kwargs["mat2"]
    del modified_sample.kwargs["mat2"]
    modified_sample.kwargs["other"] = other
    return unbind_reference(matmul_op, modified_sample)


def sample_inputs_matmul(
    op_info, device, dtype, requires_grad, op_kwargs=None, **kwargs
):
    # also run bmm samples through
    for sample_input in sample_inputs_bmm(op_info, device, dtype, requires_grad):
        # change arg name from mat2 -> other
        other = sample_input.kwargs["mat2"]
        del sample_input.kwargs["mat2"]
        sample_input.kwargs["other"] = other
        yield sample_input

    # 3D cases not covered by bmm
    for njt_3d in _sample_njts(
        device=device, dtype=dtype, requires_grad=requires_grad, dims=[3]
    ):
        # (B, j1, D) x (D, E) => (B, j1, E)
        D = njt_3d.shape[-1]
        E = D + 2
        yield SampleInput(
            njt_3d.clone().detach(),
            kwargs={"other": torch.randn(D, E, device=device, dtype=dtype)},
        )

    # 4D cases
    for njt_4d in _sample_njts(
        device=device, dtype=dtype, requires_grad=requires_grad, dims=[4]
    ):
        # (B, j1, D, E) x (E, F) => (B, j1, D, F)
        E = njt_4d.shape[-1]
        F = E + 2
        yield SampleInput(
            njt_4d.clone().detach(),
            kwargs={"other": torch.randn(E, F, device=device, dtype=dtype)},
        )

        # TODO (need factory functions):
        # (B, j1, D, E) x (B, j1, E, F) => (B, j1, D, F)


def sample_inputs_masked_select(
    op_info, device, dtype, requires_grad, op_kwargs=None, **kwargs
):
    for njt in _sample_njts(
        device=device, dtype=dtype, requires_grad=requires_grad, dims=[2]
    ):
        yield SampleInput(
            njt, kwargs={"mask": (torch.randn_like(njt, requires_grad=False) < 0.0)}
        )


def sample_inputs_nn_functional_embedding(
    op_info, device, dtype, requires_grad, **kwargs
):
    indices = torch.nested.nested_tensor(
        [
            torch.tensor([0, 2, 1, 3]),
            torch.tensor([4, 2, 1]),
            torch.tensor([6, 7, 5, 2, 4]),
        ],
        layout=torch.jagged,
        dtype=torch.int64,
        device=device,
    )

    NUM_EMBEDDINGS = 20
    EMBEDDING_DIM = 32
    weight = torch.randn(NUM_EMBEDDINGS, EMBEDDING_DIM, device=device, dtype=dtype)

    # NB: the OpInfo entry for embedding_bag expects weight first so the gradients
    # can be checked
    yield SampleInput(
        weight.clone().detach().requires_grad_(),
        args=(indices,),
    )

    yield SampleInput(
        weight.clone().detach().requires_grad_(),
        args=(indices,),
        kwargs={"padding_idx": 1},
    )


def sample_inputs_index_put(
    op_info, device, dtype, requires_grad, op_kwargs=None, **kwargs
):
    for njt in _sample_njts(
        device=device, dtype=dtype, requires_grad=requires_grad, dims=[2, 3, 4]
    ):
        for dim in range(njt.dim()):
            indices = [
                torch.tensor(list(range(njt.size(0))), device=njt.device),
                *[
                    torch.tensor([0] * njt.size(0), device=njt.device)
                    for _ in range(dim - 1)
                ],
            ]
            yield SampleInput(
                njt.clone().detach(),
                kwargs={
                    "indices": indices,
                    "values": torch.tensor(1.0, device=njt.device),
                },
            )

    # Non-cont NJT for completeness
    offsets = torch.tensor([0, 2, 5, 7], device=device)
    lengths = torch.tensor([2, 2, 2], device=device)
    indices = [
        torch.tensor([0, 1, 2], device=device),
        torch.tensor([0, 1, 1], device=device),
        torch.tensor([0, 0, 0], device=device),
    ]
    a = torch.nested.nested_tensor_from_jagged(
        torch.zeros(7, 3, device=device), offsets, lengths
    )

    yield SampleInput(
        a.clone().detach(),
        kwargs={"indices": indices, "values": torch.tensor(1.0, device=a.device)},
    )


def sample_inputs_nn_functional_embedding_bag(
    op_info, device, dtype, requires_grad, **kwargs
):
    for generate_per_sample_weight in (True, False):
        for mode in ("sum", "mean", "max"):
            # per_sample_weights is only supported for mode='sum'
            if mode != "sum" and generate_per_sample_weight:
                continue

            NUM_EMBEDDINGS = 10
            EMBEDDING_DIM = 32
            weight = torch.randn(
                NUM_EMBEDDINGS, EMBEDDING_DIM, dtype=dtype, device=device
            )

            njt = torch.nested.nested_tensor(
                [
                    torch.randint(0, NUM_EMBEDDINGS, size=(2,)),
                    torch.randint(0, NUM_EMBEDDINGS, size=(3,)),
                    torch.randint(0, NUM_EMBEDDINGS, size=(4,)),
                ],
                layout=torch.jagged,
                dtype=torch.int64,
                device=device,
            )

            per_sample_weights = None
            if generate_per_sample_weight:
                per_sample_weights = torch.randn_like(njt, dtype=dtype)

            # NB: the OpInfo entry for embedding_bag expects weight first so the gradients
            # can be checked
            yield SampleInput(
                weight,
                args=(njt,),
                kwargs={
                    "mode": mode,
                    "per_sample_weights": per_sample_weights,
                },
            )


def reference_nn_functional_embedding_bag(op, sample):
    # run reference on a single bag at a time
    new_kwargs = dict(sample.kwargs)
    new_kwargs.update(
        {"offsets": torch.tensor([0], dtype=torch.int64, device=sample.input.device)}
    )
    # flip input / weight back to what unbind_reference() expects
    sample = SampleInput(sample.args[0], args=(sample.input,), kwargs=new_kwargs)
    old_op = op.op
    op.op = torch.nn.functional.embedding_bag
    output = unbind_reference(op, sample, wrap_output_as_njt=False)
    op.op = old_op
    # concat bag outputs to get final output
    return torch.cat(output, dim=0)


def sample_inputs_nn_functional_linear(op_info, device, dtype, requires_grad, **kwargs):
    for njt in _sample_njts(
        device=device, dtype=dtype, requires_grad=requires_grad, dims=[3, 4, 5]
    ):
        # with bias
        NUM_OUTPUT = 10
        weight = torch.randn(
            NUM_OUTPUT,
            njt.size(-1),
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
        )
        bias = torch.randn(
            NUM_OUTPUT, device=device, dtype=dtype, requires_grad=requires_grad
        )
        yield SampleInput(
            njt,
            kwargs={
                "weight": weight,
                "bias": bias,
            },
        )

        # without bias
        yield SampleInput(
            njt,
            kwargs={
                "weight": weight,
            },
        )


def sample_inputs_nn_functional_rms_norm(
    op_info, device, dtype, requires_grad, **kwargs
):
    for njt in _sample_njts(
        device=device, dtype=dtype, requires_grad=requires_grad, dims=[3, 4]
    ):
        # normalize over non-ragged dims
        for start_dim in range(2, njt.dim()):
            normalized_shape = njt.shape[start_dim:]
            weight = torch.randn(
                normalized_shape,
                device=device,
                dtype=dtype,
                requires_grad=requires_grad,
            )

            yield SampleInput(
                njt,
                kwargs={
                    "normalized_shape": normalized_shape,
                    "weight": weight,
                },
            )


sample_inputs_nn_functional_threshold = partial(
    sample_inputs_elementwise_njt_unary,
    op_kwargs={"threshold": float.fromhex("0x1.3ap-3"), "value": -9},
)
# === END OP-SPECIFIC SAMPLE INPUTS FUNCS ===


# Mapping of OpInfo full names -> sample_inputs_funcs, which define the set of sample inputs
# (involving NJTs) to pass to the op. Full name consists of the OpInfo's name and variant name
# separated by a period (e.g. special.polygamma.special_polygamma_n_0). These are necessary
# to specify if they cannot be auto-generated for some reason. Try to keep these sorted
# in alphabetical order!
# === TODO: Update sample inputs funcs to respect the dim_type / contiguity kwargs! ===
njt_sample_inputs = {
    "argmax": partial(sample_inputs_njt_reduction, supports_dimlist=False),
    "argmin": partial(sample_inputs_njt_reduction, supports_dimlist=False),
    "bmm": sample_inputs_bmm,
    "clone": sample_inputs_clone,
    **{f"mvlgamma.mvlgamma_p_{p}": sample_inputs_mvl_gamma(p=1) for p in (1, 3, 5)},
    "nn.functional.embedding": sample_inputs_nn_functional_embedding,
    "nn.functional.embedding_bag": sample_inputs_nn_functional_embedding_bag,
    "nn.functional.linear": sample_inputs_nn_functional_linear,
    "nn.functional.rms_norm": sample_inputs_nn_functional_rms_norm,
    "nn.functional.threshold": sample_inputs_nn_functional_threshold,
    **{f"polygamma.polygamma_n_{n}": sample_inputs_polygamma_n(n=n) for n in range(5)},
    "special.polygamma.special_polygamma_n_0": sample_inputs_special_polygamma_n(n=0),
    "to": sample_inputs_to,
    "matmul": sample_inputs_matmul,
    "masked_select": sample_inputs_masked_select,
    "index_put": sample_inputs_index_put,
    "max.reduction_with_dim": partial(
        sample_inputs_njt_reduction, supports_dimlist=False
    ),
    "min.reduction_with_dim": partial(
        sample_inputs_njt_reduction, supports_dimlist=False
    ),
    "prod": partial(sample_inputs_njt_reduction, supports_dimlist=False),
}

njt_references = {
    "argmax": reduction_reference,
    "argmin": reduction_reference,
    "bmm": reference_bmm,
    "max.reduction_with_dim": reduction_reference,
    "min.reduction_with_dim": reduction_reference,
    "nn.functional.embedding_bag": reference_nn_functional_embedding_bag,
    "prod": reduction_reference,
}


# Mapping of OpInfo full names -> extra data to tack onto the OpInfo entry for use
# in test generation.
extra_op_data = {
    "_segment_reduce.lengths": ExtraOpData(dim_args=[["axis0"]]),
    "_segment_reduce.offsets": ExtraOpData(dim_args=[["axis0"]]),
    "all": ExtraOpData(dim_args=[["dim"], ["dim", "..."]]),
    "any": ExtraOpData(dim_args=[["dim"], ["dim", "..."]]),
    "argsort": ExtraOpData(dim_args=[["dim"]]),
    "broadcast_to": ExtraOpData(is_view=True),
    "cat": ExtraOpData(dim_args=[["dim"]]),
    "chunk": ExtraOpData(is_view=True, dim_args=[["dim"]]),
    "conj": ExtraOpData(is_view=True),
    "contiguous": ExtraOpData(is_view=True),
    "cummax": ExtraOpData(dim_args=[["dim"]]),
    "cummin": ExtraOpData(dim_args=[["dim"]]),
    "cumprod": ExtraOpData(dim_args=[["dim"]]),
    "cumsum": ExtraOpData(dim_args=[["dim"]]),
    "cumulative_trapezoid": ExtraOpData(dim_args=[["dim"]]),
    "diag_embed": ExtraOpData(dim_args=[["dim1", "dim2"]]),
    "diagonal": ExtraOpData(is_view=True, dim_args=[["dim1", "dim2"]]),
    "diagonal_copy": ExtraOpData(dim_args=[["dim1", "dim2"]]),
    "diagonal_scatter": ExtraOpData(dim_args=[["dim1", "dim2"]]),
    "diff": ExtraOpData(dim_args=[["dim"]]),
    "expand": ExtraOpData(is_view=True),
    "expand_as": ExtraOpData(is_view=True),
    "fft.fft": ExtraOpData(dim_args=[["dim"]]),
    "fft.hfft": ExtraOpData(dim_args=[["dim"]]),
    "fft.ifft": ExtraOpData(dim_args=[["dim"]]),
    "fft.ihfft": ExtraOpData(dim_args=[["dim"]]),
    "fft.irfft": ExtraOpData(dim_args=[["dim"]]),
    "fft.rfft": ExtraOpData(dim_args=[["dim"]]),
    "flatten": ExtraOpData(is_view=True, dim_args=[["start_dim", "end_dim"]]),
    "flip": ExtraOpData(dim_args=[["dims", "..."]]),
    "gather": ExtraOpData(dim_args=[["dim"]]),
    "imag": ExtraOpData(is_view=True),
    "index_add": ExtraOpData(dim_args=[["dim"]]),
    "index_copy": ExtraOpData(dim_args=[["dim"]]),
    "index_fill": ExtraOpData(dim_args=[["dim"]]),
    "index_reduce.amax": ExtraOpData(dim_args=[["dim"]]),
    "index_reduce.amin": ExtraOpData(dim_args=[["dim"]]),
    "index_reduce.mean": ExtraOpData(dim_args=[["dim"]]),
    "index_reduce.prod": ExtraOpData(dim_args=[["dim"]]),
    "index_select": ExtraOpData(dim_args=[["dim"]]),
    "kthvalue": ExtraOpData(dim_args=[["dim"]]),
    "linalg.cross": ExtraOpData(dim_args=[["dim"]]),
    "linalg.diagonal": ExtraOpData(is_view=True, dim_args=[["dim1", "dim2"]]),
    "linalg.tensorsolve": ExtraOpData(dim_args=[["dims", "..."]]),
    "linalg.vecdot": ExtraOpData(dim_args=[["dim"]]),
    "log_softmax": ExtraOpData(dim_args=[["dim"]]),
    "logcumsumexp": ExtraOpData(dim_args=[["dim"]]),
    "max.reduction_with_dim": ExtraOpData(dim_args=[["dim"]]),
    "median": ExtraOpData(dim_args=[["dim"]]),
    "min.reduction_with_dim": ExtraOpData(dim_args=[["dim"]]),
    "mode": ExtraOpData(dim_args=[["dim"]]),
    "movedim": ExtraOpData(
        dim_args=[["source", "destination"], ["source", "...", "destination", "..."]]
    ),
    "nanmedian": ExtraOpData(dim_args=[["dim"]]),
    "narrow": ExtraOpData(is_view=True, dim_args=[["dim"]]),
    "narrow_copy": ExtraOpData(dim_args=[["dim"]]),
    "nn.functional.cosine_similarity": ExtraOpData(dim_args=[["dim"]]),
    "nn.functional.glu": ExtraOpData(dim_args=[["dim"]]),
    "permute": ExtraOpData(is_view=True, dim_args=[["dims", "..."]]),
    "positive": ExtraOpData(is_view=True),
    "prod": ExtraOpData(dim_args=[["dim"]]),
    "ravel": ExtraOpData(is_view=True),
    "real": ExtraOpData(is_view=True),
    "renorm": ExtraOpData(dim_args=[["dim"]]),
    "reshape": ExtraOpData(is_view=True),
    "reshape_as": ExtraOpData(is_view=True),
    "roll": ExtraOpData(dim_args=[["dims", "..."]]),
    "rot90": ExtraOpData(dim_args=[["dims", "..."]]),
    "scatter": ExtraOpData(dim_args=[["dim"]]),
    "scatter_add": ExtraOpData(dim_args=[["dim"]]),
    "scatter_reduce.amax": ExtraOpData(dim_args=[["dim"]]),
    "scatter_reduce.amin": ExtraOpData(dim_args=[["dim"]]),
    "scatter_reduce.mean": ExtraOpData(dim_args=[["dim"]]),
    "scatter_reduce.prod": ExtraOpData(dim_args=[["dim"]]),
    "scatter_reduce.sum": ExtraOpData(dim_args=[["dim"]]),
    "select": ExtraOpData(is_view=True, dim_args=[["dim"]]),
    "select_scatter": ExtraOpData(dim_args=[["dim"]]),
    "slice": ExtraOpData(is_view=True, dim_args=[["dim"]]),
    "slice_scatter": ExtraOpData(dim_args=[["dim"]]),
    "softmax": ExtraOpData(dim_args=[["dim"]]),
    "sort": ExtraOpData(dim_args=[["dim"]]),
    "split": ExtraOpData(is_view=True, dim_args=[["dim"]]),
    "split_with_sizes": ExtraOpData(is_view=True, dim_args=[["dim"]]),
    "split_with_sizes_copy": ExtraOpData(dim_args=[["dim"]]),
    "squeeze": ExtraOpData(is_view=True, dim_args=[["dim"], ["dim", "..."]]),
    "squeeze_copy": ExtraOpData(dim_args=[["dim"], ["dim", "..."]]),
    "stack": ExtraOpData(dim_args=[["dim"]]),
    "t": ExtraOpData(is_view=True),
    "tensor_split": ExtraOpData(is_view=True, dim_args=[["dim"]]),
    "tensordot": ExtraOpData(dim_args=[["dims", "..."]]),
    "tile": ExtraOpData(dim_args=[["dims", "..."]]),
    "topk": ExtraOpData(dim_args=[["dim"]]),
    "transpose": ExtraOpData(is_view=True, dim_args=[["dim0", "dim1"]]),
    "transpose_copy": ExtraOpData(dim_args=[["dim0", "dim1"]]),
    "trapezoid": ExtraOpData(dim_args=[["dim"]]),
    "trapz": ExtraOpData(dim_args=[["dim"]]),
    "unbind": ExtraOpData(is_view=True, dim_args=[["dim"]]),
    "unflatten": ExtraOpData(is_view=True, dim_args=[["dim"]]),
    "unfold": ExtraOpData(is_view=True, dim_args=[["dimension"]]),
    "unfold_copy": ExtraOpData(dim_args=[["dimension"]]),
    "unsafe_chunk": ExtraOpData(dim_args=[["dim"]]),
    "unsafe_split": ExtraOpData(dim_args=[["dim"]]),
    "unsqueeze": ExtraOpData(is_view=True, dim_args=[["dim"]]),
    "unsqueeze_copy": ExtraOpData(dim_args=[["dim"]]),
    "view": ExtraOpData(is_view=True),
    "view_as": ExtraOpData(is_view=True),
    "view_as_complex": ExtraOpData(is_view=True),
    "view_as_real": ExtraOpData(is_view=True),
}


# Translates an OpInfo entry to one that operates on NJTs.
def translate_opinfo(op):
    new_op = copy(op)
    new_op.supports_njt = True
    # add some extra info for use in generating tests on the right subset of ops
    new_op._extra_op_data = extra_op_data.get(op.full_name, ExtraOpData())

    if op.full_name in njt_sample_inputs:
        new_op.sample_inputs_func = njt_sample_inputs[op.full_name]
        new_op.ref = njt_references.get(op.full_name, unbind_reference)
    elif isinstance(op, UnaryUfuncInfo):
        new_op.sample_inputs_func = partial(
            sample_inputs_elementwise_njt_unary, op_kwargs=None
        )
        new_op.ref = unbind_reference
    elif isinstance(op, BinaryUfuncInfo):
        new_op.sample_inputs_func = partial(
            sample_inputs_elementwise_njt_binary, op_kwargs=None
        )
        new_op.ref = unbind_reference
    elif isinstance(op, ReductionOpInfo):
        new_op.sample_inputs_func = partial(sample_inputs_njt_reduction, op_kwargs=None)
        new_op.ref = reduction_reference
    # TODO: Translate the rest of the OpInfos
    else:
        new_op.sample_inputs_func = unsupported_sample_inputs_func(op.full_name)
        new_op.ref = unsupported_reference(op.full_name)
        new_op.supports_njt = False

    return new_op


njt_op_db = [translate_opinfo(op) for op in op_db]
