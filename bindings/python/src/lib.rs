use pyo3::prelude::*;

#[pyfunction]
fn say_hello() -> PyResult<String> {
    Ok(tokengeex::say_hello())
}

#[pymodule]
#[pyo3(name = "tokengeex")]
fn tokengeex_module(_: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(say_hello, m)?)?;

    Ok(())
}
