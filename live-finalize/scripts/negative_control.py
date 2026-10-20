import importlib, sys, pytest
module = importlib.import_module("flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim.hopper_mxfp4")
assert module.__file__.startswith("/test/source/"), module.__file__
print("NEGATIVE_CONTROL_BASELINE_MODULE", module.__file__, flush=True)
sys.exit(pytest.main(["-q", "/test/candidate-source/tests/moe_ep/test_sm90_pull_mxfp4_kernel_vs_reference.py::test_sm90_mxfp4_kernel_matches_independent_reference[1]"]))
