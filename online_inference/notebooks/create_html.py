import os
import click


@click.command(name="ipynb_to_html")
@click.argument("ipynb_path")
def ipynb_to_html_command(ipynb_path: str):
    os.system('jupyter nbconvert --to html ' + ipynb_path)


if __name__ == "__main__":
    ipynb_to_html_command()
