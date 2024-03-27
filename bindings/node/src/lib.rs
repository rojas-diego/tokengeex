use std::str::FromStr;

use neon::{prelude::*, types::buffer::TypedArray};
use tokengeex::Tokenizer;

struct JsTokenizer {
    tokenizer: Tokenizer,
}

impl Finalize for JsTokenizer {}

impl JsTokenizer {
    fn js_from_str(mut cx: FunctionContext) -> JsResult<JsBox<JsTokenizer>> {
        let serialized = cx.argument::<JsString>(0)?.value(&mut cx);

        match Tokenizer::from_str(&serialized) {
            Ok(tokenizer) => Ok(cx.boxed(JsTokenizer { tokenizer })),
            Err(e) => cx.throw_error(e.to_string()),
        }
    }

    fn js_encode(mut cx: FunctionContext) -> JsResult<JsArray> {
        let this = cx.argument::<JsBox<JsTokenizer>>(0)?;
        let input = cx.argument::<JsString>(1)?.value(&mut cx);

        let tokens = match this.tokenizer.encode(&input) {
            Ok(tokens) => tokens,
            Err(e) => return cx.throw_error(e.to_string()),
        };

        let js_tokens = JsArray::new(&mut cx, tokens.len());
        for (i, token) in tokens.iter().enumerate() {
            let js_token = cx.number(*token);
            js_tokens.set(&mut cx, i as u32, js_token)?;
        }

        Ok(js_tokens)
    }

    fn js_decode(mut cx: FunctionContext) -> JsResult<JsString> {
        let this = cx.argument::<JsBox<JsTokenizer>>(0)?;
        let tokens = cx.argument::<JsArray>(1)?;
        let tokens = tokens.to_vec(&mut cx)?;
        let include_special_tokens = cx.argument::<JsBoolean>(2)?.value(&mut cx);

        let token_ids: Vec<u32> = tokens
            .iter()
            .map(|token| {
                token
                    .downcast::<JsNumber, _>(&mut cx)
                    .unwrap()
                    .value(&mut cx) as u32
            })
            .collect();

        let decoded = match this.tokenizer.decode(&token_ids, include_special_tokens) {
            Ok(decoded) => decoded,
            Err(e) => return cx.throw_error(e.to_string()),
        };

        Ok(cx.string(&decoded))
    }

    fn js_token_to_id(mut cx: FunctionContext) -> JsResult<JsNumber> {
        let this = cx.argument::<JsBox<JsTokenizer>>(0)?;
        let token = cx.argument::<JsBuffer>(1)?;
        let token = token.as_slice(&cx);

        let token_id = match this.tokenizer.token_to_id(token.to_vec()) {
            Some(token_id) => token_id,
            None => return cx.throw_error("Token not found"),
        };

        Ok(cx.number(token_id))
    }

    fn js_id_to_token(mut cx: FunctionContext) -> JsResult<JsObject> {
        let this = cx.argument::<JsBox<JsTokenizer>>(0)?;
        let token_id = cx.argument::<JsNumber>(1)?.value(&mut cx) as u32;

        let token = match this.tokenizer.id_to_token(token_id) {
            Some(token) => token,
            None => return cx.throw_error("Token not found"),
        };

        let (token, score) = token;

        let token = JsBuffer::from_slice(&mut cx, token.as_slice())?;
        let score = cx.number(score);

        let obj = JsObject::new(&mut cx);
        obj.set(&mut cx, "token", token)?;
        obj.set(&mut cx, "score", score)?;

        Ok(obj)
    }
}

#[neon::main]
fn main(mut cx: ModuleContext) -> NeonResult<()> {
    cx.export_function("fromString", JsTokenizer::js_from_str)?;
    cx.export_function("encode", JsTokenizer::js_encode)?;
    cx.export_function("decode", JsTokenizer::js_decode)?;
    cx.export_function("tokenToId", JsTokenizer::js_token_to_id)?;
    cx.export_function("idToToken", JsTokenizer::js_id_to_token)?;

    Ok(())
}
