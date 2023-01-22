
class Config:
    __config_dict = {}
    __section_chars = ('[', ']')

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.parse_file(self.file_path)

    def parse_file(self, file_path: str) -> None:
        with open(file_path, "r") as file:
            for line in file:
                if line[0] == '[':
                    for char in self.__section_chars:
                        line.replace(char, "")
                    self.__config_dict[line] = {}
                key, value = line.strip().split("=")
                self.__config_dict[key] = value

#TODO:
#   - handle different types
#   - create new ConfigObject??
#   -     config_path = os.path.join(local_dir, "config_feedforward.txt")
