use pyo3::create_exception;
use pyo3::prelude::*;
use pyo3::types::*;

create_exception!(tokengeex, TokenGeeXError, pyo3::exceptions::PyException);

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

struct PyTokenGeeXError {
    inner: tokengeex::Error,
}

impl From<tokengeex::Error> for PyTokenGeeXError {
    fn from(e: tokengeex::Error) -> Self {
        PyTokenGeeXError { inner: e }
    }
}

impl From<PyTokenGeeXError> for PyErr {
    fn from(e: PyTokenGeeXError) -> PyErr {
        TokenGeeXError::new_err(e.inner.to_string())
    }
}

#[pymethods]
impl PyTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>, PyTokenGeeXError> {
        self.tokenizer.encode(text).map_err(|e| e.into())
    }

    fn decode(&self, ids: Vec<u32>) -> Result<String, PyTokenGeeXError> {
        self.tokenizer.decode(&ids).map_err(|e| e.into())
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
fn tokengeex_load_py(filename: &str) -> Result<PyTokenizer, PyTokenGeeXError> {
    let tokenizer =
        tokengeex::load(filename).map_err(std::convert::Into::<PyTokenGeeXError>::into)?;
    Ok(PyTokenizer::from(tokenizer))
}

#[pymodule]
#[pyo3(name = "tokengeex")]
fn tokengeex_module(_: Python, m: &PyModule) -> PyResult<()> {
    // Module: TokenGeeX
    m.add_function(wrap_pyfunction!(tokengeex_load_py, m)?)?;
    m.add_class::<PyTokenizer>()?;

    Ok(())
}
