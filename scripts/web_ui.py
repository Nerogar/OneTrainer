from util.import_util import script_imports

script_imports()

from modules.web.api_server import run


def main():
    run()


if __name__ == "__main__":
    main()
