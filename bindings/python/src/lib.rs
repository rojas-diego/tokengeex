use pyo3::prelude::*;

#[pyfunction]
#[pyo3(name = "encode")]
fn tokengeex_capcode_encode_py(input: &str) -> PyResult<String> {
    Ok(tokengeex::capcode::encode(input))
}

#[pyfunction]
#[pyo3(name = "decode")]
fn tokengeex_capcode_decode_py(input: &str) -> PyResult<String> {
    Ok(tokengeex::capcode::decode(input))
}

#[pyfunction]
#[pyo3(name = "is_marker")]
fn tokengeex_capcode_is_marker_py(c: char) -> PyResult<bool> {
    Ok(tokengeex::capcode::is_marker(c))
}

#[pymodule]
#[pyo3(name = "tokengeex")]
fn tokengeex_module(py: Python, m: &PyModule) -> PyResult<()> {
    // Submodule: Capcode
    let capcode = PyModule::new(py, "capcode")?;
    capcode.add_function(wrap_pyfunction!(tokengeex_capcode_encode_py, capcode)?)?;
    capcode.add_function(wrap_pyfunction!(tokengeex_capcode_decode_py, capcode)?)?;
    capcode.add_function(wrap_pyfunction!(tokengeex_capcode_is_marker_py, capcode)?)?;

    m.add_submodule(capcode)?;

    Ok(())
}
