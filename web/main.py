import gradio as gr
import tokengeex


def tokengeex_demo(vocab: str, input: str, lang: str):
    if lang != "none":
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
    tokens = [tokenizer.id_to_token(id) for id in ids]  # type: ignore
    tokens = list(filter(lambda token: token is not None, tokens))
    tokens = [token[0].decode(encoding="utf-8", errors="ignore") for token in tokens]  # type: ignore

    categories = [str(i) for i in range(1, 10)]

    tokens = [(token, categories[id % 9]) for id, token in enumerate(tokens)]

    return (
        gr.HighlightedText(
            value=tokens,  # type: ignore
            adjacent_separator="",
            combine_adjacent=True,
            show_label=False,
            show_legend=False,
        ),
        gr.Text(
            value="The 'D' marker signifies the next space is deleted and the 'U' and 'C' markers indicate the next word is uppercase and capitalized, respectively.",
        ),
    )


demo = gr.Interface(
    fn=tokengeex_demo,
    inputs=[
        gr.Dropdown(
            choices=["capcode-65k.json", "capcode-131k.json"],
            value="capcode-65k.json",
            label="Vocabulary File",
        ),
        gr.Code(
            lines=5,
            label="Code",
        ),
        gr.Dropdown(
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
        ),
    ],  # type: ignore
    outputs=[
        "highlight",
        "text",
    ],
)

demo.launch()  # Share your demo with just 1 extra parameter 🚀
