use pyo3::prelude::*;
use pyo3::types::*;

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
    /// Encode a string to a list of token IDs.
    #[pyo3(text_signature = "(self, text)")]
    fn encode(&self, text: &str) -> Vec<u32> {
        self.tokenizer.encode(text)
    }

    /// Decode a list of token IDs to a string.
    #[pyo3(text_signature = "(self, ids)")]
    fn decode(&self, ids: Vec<u32>) -> String {
        self.tokenizer.decode(&ids)
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

    /// Used to pickle the Tokenizer. Useful for sharing a tokenizer between
    /// Python processes.
    fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        let data = serde_json::to_string(&self.tokenizer).map_err(|e| {
            pyo3::exceptions::PyException::new_err(format!(
                "Error while attempting to pickle Tokenizer: {}",
                e
            ))
        })?;
        Ok(PyBytes::new(py, data.as_bytes()).to_object(py))
    }

    /// Used to unpickle the Tokenizer. Useful for sharing a tokenizer between
    /// Python processes.
    fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        match state.extract::<&PyBytes>(py) {
            Ok(s) => {
                self.tokenizer = serde_json::from_slice(s.as_bytes()).map_err(|e| {
                    pyo3::exceptions::PyException::new_err(format!(
                        "Error while attempting to unpickle Tokenizer: {}",
                        e
                    ))
                })?;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }
}

#[pyfunction]
#[pyo3(name = "load")]
fn tokengeex_load_py(filename: &str) -> PyResult<PyTokenizer> {
    Ok(PyTokenizer::from(tokengeex::load(filename).unwrap()))
}

#[pymodule]
#[pyo3(name = "tokengeex")]
fn tokengeex_module(_: Python, m: &PyModule) -> PyResult<()> {
    // Module: TokenGeeX
    m.add_function(wrap_pyfunction!(tokengeex_load_py, m)?)?;

    Ok(())
}
