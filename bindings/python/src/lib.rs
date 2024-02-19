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

#[pyclass(dict, module = "tokengeex", name = "Tokenizer")]
#[derive(Clone)]
struct PyTokenizer {
    tokenizer: tokengeex::Tokenizer,
}

impl PyTokenizer {
    fn from(tokenizer: tokengeex::Tokenizer) -> Self {
        PyTokenizer { tokenizer }
    }
}

#[pymethods]
impl PyTokenizer {
    fn encode(&self, input: &str) -> Vec<u32> {
        self.tokenizer.encode(input)
    }

    fn decode(&self, input: Vec<u32>) -> String {
        self.tokenizer.decode(&input)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.tokenizer.token_to_id(token)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.tokenizer.id_to_token(id)
    }

    fn vocab_size(&self) -> usize {
        self.tokenizer.vocab_size()
    }
}

#[pyfunction]
#[pyo3(name = "load")]
fn tokengeex_load_py(filename: &str) -> PyResult<PyTokenizer> {
    Ok(PyTokenizer::from(tokengeex::load(filename).unwrap()))
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

    // Module: TokenGeeX
    m.add_function(wrap_pyfunction!(tokengeex_load_py, m)?)?;

    Ok(())
}
