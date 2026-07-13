from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def mini_psd_path() -> Path:
    return FIXTURES / "mini_model.psd"


@pytest.fixture(scope="session")
def mini_rig():
    from raig.compiler.compile import compile_psd

    return compile_psd(FIXTURES / "mini_model.psd")


@pytest.fixture(scope="session")
def gl_ctx():
    import moderngl

    try:
        ctx = moderngl.create_context(standalone=True)
    except Exception as e:  # no GL on this machine/CI
        pytest.skip(f"no standalone GL context: {e}")
    yield ctx
    ctx.release()
