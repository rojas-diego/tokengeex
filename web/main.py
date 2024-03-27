import html

import gradio as gr
import tokengeex


def split_and_keep_sep(s, sep):
    parts = []
    current = []
    for char in s:
        if char in sep:
            if current:
                parts.append("".join(current))
                current = []
            parts.append(char)
        else:
            current.append(char)
    if current:  # Add the last part if there's any
        parts.append("".join(current))
    return parts


def make_element(i, id, token, score):
    colors = [
        "rgba(239,65,70,.2)",
        "rgba(39, 181, 234, 0.2)",
        "rgba(107,64,216,.2)",
        "rgba(104,222,122,.2)",
        "rgba(244,172,54,.2)",
    ]

    color = colors[i % len(colors)]
    splits = split_and_keep_sep(token, ["\n", "\t"])
    element = ""

    for split in splits:
        if split == "\n":
            element += f'<span style="background-color: {color}; white-space: nowrap;">↵</span><br>'
        elif split == "\t":
            element += f'<span style="background-color: {color}; white-space: nowrap;">····</span>'
        else:
            while split.startswith(" "):
                element += f'<span style="background-color: {color}; white-space: nowrap;">·</span>'
                split = split[1:]
            after = ""
            while split.endswith(" "):
                after += f'<span style="background-color: {color}; white-space: nowrap;">·</span>'
                split = split[:-1]

            escaped = html.escape(split)
            element += f'<span style="background-color: {color}; white-space: nowrap;">{escaped}</span>'
            element += after

    return element


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            drop_down_vocab = gr.Dropdown(
                choices=[
                    "base-131k.json",
                    "capcode-131k.json",
                    "capcode-131k-extended.json",
                    "capcode-65k.json",
                    "capcode-65k-extended.json",
                    # "base-65k.json",
                ],
                value="base-131k.json",
                label="Vocabulary File",
            )
            code = gr.Code(
                lines=5,
                label="Code",
                value="",
            )
            drop_down_lang = gr.Dropdown(
                label="Example",
                choices=[
                    "none",
                    "assembly",
                    "cmake",
                    "dart",
                    "go",
                    "infilling",
                    "jsx",
                    "makefile",
                    "php",
                    "ruby",
                    "sql",
                    "typescript",
                    "zig",
                    "c-sharp",
                    "cpp",
                    "diff",
                    "haskell",
                    "java",
                    "julia",
                    "markdown",
                    "powershell",
                    "rust",
                    "swift",
                    "vue",
                    "c",
                    "css",
                    "dockerfile",
                    "hcl",
                    "javascript",
                    "kotlin",
                    "pascal",
                    "python",
                    "scala",
                    "tex",
                    "xml",
                    "chinese-markdown",
                    "cuda",
                    "elixir",
                    "html",
                    "json",
                    "lua",
                    "perl",
                    "r",
                    "shell",
                    "toml",
                    "yaml",
                ],
            )

            submit_button = gr.Button("Submit")

    with gr.Row():
        out_num_tokens = gr.Number(
            label="Number of Tokens",
        )
        out_characters_per_token = gr.Number(
            label="Characters per Token",
        )

    with gr.Row():
        out_html = gr.HTML(
            value="",
        )
        output_text = gr.Code()

    def submit(vocab: str, input: str, lang: str):
        if lang and lang != "none" and input == "":
            with open(f"./hub/data/test/{lang}.bin", "rb") as f:
                # Read until the first 0x00 byte
                buff = bytearray()
                while True:
                    read = f.read(4096)
                    if not read:
                        break
                    if b"\x00" in read:
                        buff += read[: read.index(b"\x00")]
                        break
                    buff += read
                input = buff.decode("utf-8", errors="ignore")

        tokenizer = tokengeex.load(f"./hub/vocab/{vocab}")
        ids = tokenizer.encode(input)
        tokens = [(id, tokenizer.id_to_token(id)) for id in ids]  # type: ignore
        tokens = list(filter(lambda token: token[1] is not None, tokens))
        tokens = [
            (id, token[0].decode(encoding="utf-8", errors="ignore"), token[1])  # type: ignore
            for (id, token) in tokens
        ]  # type: ignore

        num_chars = len(input)
        num_tokens = len(tokens)
        characters_per_token = num_chars / num_tokens
        characters_per_token = round(characters_per_token, 2)

        # Transform each token into an HTML element
        tokens = [
            make_element(i, id, token, score)  # type: ignore
            for (i, (id, token, score)) in enumerate(tokens)
        ]

        html = f'<code style="background-color: white; font-family: monospace; overflow: scroll"><pre style="overflow: scroll;">{"".join(tokens)}</pre></code>'

        return (
            gr.Number(
                value=num_tokens,
            ),
            gr.Number(
                value=characters_per_token,
            ),
            gr.HTML(
                value=html,
            ),
            gr.Code(
                value=input,
            ),
        )

    submit_button.click(
        submit,
        inputs=[drop_down_vocab, code, drop_down_lang],
        outputs=[out_num_tokens, out_characters_per_token, out_html, output_text],
    )

demo.launch()  # Share your demo with just 1 extra parameter 🚀
