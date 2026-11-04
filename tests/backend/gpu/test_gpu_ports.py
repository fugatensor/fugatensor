"""Real native CUDA port binding, alias identity and external owner lifetime."""
import gc
import weakref
from unittest.mock import patch

import pytest

from fugatensor._ask import CUDAProgram, ExecutionSession
from fugatensor._ask.cuda import CUDAError
from fugatensor._ask.runtime import DeviceTensor
from tests.backend.test_runtime_and_cache import ArrayOwner

pytestmark = pytest.mark.cuda


@pytest.fixture(params=(False, True), ids=("stream", "graph"))
def prepared_identity(request):
    with ExecutionSession() as session:
        plan = {
            "schema_version": 1, "kind": "ask.cuda.execution_plan",
            "inputs": ["x"], "outputs": ["x", "x"],
            "buffers": [{"id": "x", "shape": [2, 5], "element_type": "f32"}],
            "steps": [], "scheduling": {"policy": "sequential", "stream": "session"},
        }
        yield session, CUDAProgram(plan, {}).prepare(session, graph=request.param)


def borrowed_input(session):
    storage = session.from_host(list(range(10)), shape=(2, 5), dtype="f32")
    owner = ArrayOwner(**storage.__cuda_array_interface__)
    owner.storage = storage
    return owner, session.from_cuda_array(owner)


def test_prepared_run_preserves_aliases_and_retains_external_owner_until_completion(prepared_identity):
    session, prepared = prepared_identity
    owner, borrowed = borrowed_input(session)
    tracked = weakref.ref(owner)
    before = session._native.stats['external_stream_waits']
    with ExecutionSession() as producer:
        owner.description['stream'] = producer.stream
        outputs = prepared.run([borrowed])
        assert session._native.stats['external_stream_waits'] == before + 1
    assert outputs[0] is borrowed and outputs[1] is borrowed
    del outputs, borrowed, owner
    gc.collect()
    assert tracked() is not None
    session.synchronize()
    gc.collect()
    assert tracked() is None


def test_prepared_entrypoints_reject_invalid_ports_before_native_submission(prepared_identity):
    session, prepared = prepared_identity
    owner, borrowed = borrowed_input(session)
    wrong_shape, wrong_dtype = session.tensor((10,), "f32"), session.tensor((2, 5), "i32")
    before = session._native.stats
    with patch.object(DeviceTensor, "_prepare_access", side_effect=AssertionError("access before complete port validation")):
        for values in (None, [], [object()], [wrong_shape], [wrong_dtype]):
            with pytest.raises(CUDAError):
                prepared.run(values)
            with pytest.raises(CUDAError):
                prepared.submit_many(values, [borrowed, borrowed])
        for values in (None, [], [borrowed], [borrowed, object()], [borrowed, wrong_shape], [borrowed, wrong_dtype]):
            with pytest.raises(CUDAError):
                prepared.submit_many([borrowed], values)
    assert session._native.stats == before
    with ExecutionSession() as other:
        foreign = other.tensor((2, 5), "f32")
        for call in (lambda: prepared.run([foreign]), lambda: prepared.submit_many([borrowed], [borrowed, foreign])):
            with pytest.raises(CUDAError, match="different execution session"):
                call()
    owner.description["data"] = (borrowed.data_ptr, True)
    with pytest.raises(CUDAError, match="changed"):
        prepared.submit_many([borrowed], [borrowed, borrowed])
    assert session._native.stats["program_submissions"] == before["program_submissions"]


@pytest.mark.parametrize("method", ("run", "submit"))
def test_submission_retains_port_snapshot_when_caller_lists_change_during_launch(prepared_identity, method):
    session, prepared = prepared_identity
    owner, borrowed = borrowed_input(session)
    owners = [owner]
    inputs = [borrowed]
    if method == "submit":
        output_owner = ArrayOwner(**owner.description)
        output_owner.storage = owner.storage
        owners.append(output_owner)
        output = session.from_cuda_array(output_owner)
        outputs = [output, output]
        del output_owner, output
    else:
        outputs = []
    tracked = [weakref.ref(value) for value in owners]
    del owner, borrowed, owners
    prepare_access = DeviceTensor._prepare_access

    def mutate_lists(tensor, current_session):
        inputs.clear()
        outputs.clear()
        gc.collect()
        assert all(value() is not None for value in tracked)
        prepare_access(tensor, current_session)

    with patch.object(DeviceTensor, "_prepare_access", mutate_lists):
        if method == "run":
            prepared.run(inputs)
        else:
            prepared.submit_many(inputs, outputs)
    gc.collect()
    assert all(value() is not None for value in tracked)
    session.synchronize()
    gc.collect()
    assert all(value() is None for value in tracked)


def test_output_aliases_and_zero_step_identity(prepared_identity):
    session, prepared = prepared_identity
    x = session.from_host(list(range(10)), shape=(2, 5), dtype="f32")
    event = prepared.submit_many([x], [x, x])
    assert event.wait() is event and event.query()
    outputs = prepared.run([x])
    assert outputs[0] is x and outputs[1] is x
    assert x.to_host() == list(range(10))


@pytest.mark.parametrize("graph", (False, True), ids=("stream", "graph"))
def test_computed_repeated_outputs_share_one_tensor(tmp_path, graph):
    from fugatensor._ask.cuda import install_external
    from fugatensor._ask.gpu import CUDAKernel
    from tests.backend.gpu.test_gpu_execution_plan import PTX
    from tests.backend.test_cuda_compilation import make_external_artifact

    artifact = make_external_artifact(payload=PTX)
    kernel = CUDAKernel(artifact, install_external(artifact, tmp_path), {}, {})
    plan = {
        "schema_version": 1, "kind": "ask.cuda.execution_plan",
        "inputs": ["x"], "outputs": ["z", "z", "x"],
        "buffers": [{"id": name, "shape": [2, 5], "element_type": "i32"} for name in ("x", "z")],
        "steps": [{"id": "scale", "kind": "external", "function": "scale2", "external_id": "scale",
                   "inputs": ["x"], "outputs": ["z"], "depends_on": []}],
        "scheduling": {"policy": "sequential", "stream": "session"},
    }
    with ExecutionSession() as session:
        prepared = CUDAProgram(plan, {"scale": kernel}).prepare(session, graph=graph)
        x = session.from_host(list(range(10)), shape=(2, 5), dtype="i32")
        outputs = prepared.run([x])
        assert outputs[0] is outputs[1] and outputs[2] is x
        assert outputs[0].to_host() == [2 * value for value in range(10)]
        owner = ArrayOwner(**outputs[0].__cuda_array_interface__)
        owner.storage = outputs[0]
        owner.description['data'] = (outputs[0].data_ptr, True)
        readonly = session.from_cuda_array(owner)
        before = session._native.stats['program_submissions']
        for repeated in ((readonly, outputs[0]), (outputs[0], readonly)):
            with pytest.raises(CUDAError, match='read-only'):
                prepared.submit_many([x], [*repeated, x])
        assert session._native.stats['program_submissions'] == before
