from pathlib import Path
import re


def markdown2txt(
    markdown_path: str | Path,
    output_txt: str | Path | None = None,
) -> str:
    markdown_path = Path(markdown_path)
    if not markdown_path.exists():
        raise FileNotFoundError(markdown_path)

    text = markdown_path.read_text(encoding="utf-8")

    plain = markdown_to_text(text)

    if output_txt is None:
        out_dir = markdown_path.parent / "result" / "texte"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_txt = out_dir / markdown_path.with_suffix(".txt").name
    else:
        output_txt = Path(output_txt)
        output_txt.parent.mkdir(parents=True, exist_ok=True)

    output_txt.write_text(plain, encoding="utf-8")
    return str(output_txt)


def markdown_to_text(md: str) -> str:
    lines = md.splitlines()
    out = []

    for line in lines:
        l = line.rstrip()

        # Titres → saut de ligne + texte
        if m := re.match(r"(#{1,6})\s+(.*)", l):
            out.append("")
            out.append(m.group(2).strip())
            out.append("")
            continue

        # Images → garder alt
        l = re.sub(r"!\[(.*?)\]\(.*?\)", r"\1", l)

        # Liens → garder texte
        l = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", l)

        # Gras / italique / code
        l = re.sub(r"\*\*(.*?)\*\*", r"\1", l)
        l = re.sub(r"\*(.*?)\*", r"\1", l)
        l = re.sub(r"`(.*?)`", r"\1", l)

        # Listes → enlever le marqueur
        l = re.sub(r"^\s*[-*+]\s+", "", l)
        l = re.sub(r"^\s*\d+\.\s+", "", l)

        # Blockquote
        l = re.sub(r"^\s*>\s+", "", l)

        # Table Markdown → remplacer | par espace
        if "|" in l:
            l = l.replace("|", " ")

        # Nettoyage espaces
        l = re.sub(r"\s{2,}", " ", l)

        out.append(l)

    # Nettoyage final
    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


__all__ = ["markdown2txt"]
