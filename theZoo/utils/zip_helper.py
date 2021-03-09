import pyzipper


class ZipHelper:

    def __init__(self, output_dir=None, password=None):
        self.output_dir = output_dir
        self.password = password

    def extract_all(self, zip_path, output_dir=None):
        if output_dir is None:
            output_dir = self.output_dir
            print(f"extracting ... {zip_path}")
        with pyzipper.AESZipFile(zip_path, 'r') as f:
            f.extractall(pwd=self.password, path=output_dir)
