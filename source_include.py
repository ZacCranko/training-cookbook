import re
from pathlib import Path
import ast


def define_env(env):
    @env.macro
    def tagged_block(filepath, tag, lines: str | None = None):
        try:
            full_path = Path(env.project_dir) / filepath
            regex_pattern = rf"# tag: {tag}\n(.*?)\s*# tag: {tag}"
            pattern = re.compile(regex_pattern, re.DOTALL)

            if not (content := pattern.search(full_path.read_text()).group(1)):
                return f"Error: Tag '{tag}' not found in '{filepath}'"

            if lines is None:
                return content.strip()

            if lines.startswith("[") and lines.endswith("]"):
                indexer = ast.literal_eval(lines)
            elif ":" in lines:
                parts_str = (lines.split(":") + ["", "", ""])[:3]
                indexer = slice(*(int(p.strip()) if p.strip() else None for p in parts_str))
            else:
                indexer = slice(int(lines), int(lines) + 1)

            line_list = content.split("\n")[indexer]
            indent_level = next(idx for idx, chr in enumerate(line_list[0]) if not chr.isspace())

            return "\n".join(line[indent_level:] for line in line_list)

        except Exception as exc:
            return f"Error: '{exc}'"
