import os
import urllib.parse


class PrettyOutput:

    @staticmethod
    def output_per_line(text: str, words_per_line: int=10) -> str:
        text_parts = text.split('\n')
        pretty_text = ''

        for text_part in text_parts:
            words = text_part.split(' ')
            for i, word in enumerate(words):
                pretty_text += word + ' '
                if (i + 1) % words_per_line == 0 and i != len(words) - 1:
                    pretty_text += '\n'
            pretty_text += '\n'

        return pretty_text


    @staticmethod
    def pretty_output_with_context(answer: str, context: list) -> str:
        return_str = f"{answer}\n\nKontext:\n"
        for doc in context:
            file_path = doc.metadata["source"]
            formatted_path = file_path.replace("\\", "/").replace("Data", "Source")
            encoded_path = urllib.parse.quote(formatted_path)
            file_url = f"file:///{encoded_path}"
            return_str += f"- [{os.path.basename(file_path)}]({file_url}) - Seite {doc.metadata['page_number']}\n"

        return return_str