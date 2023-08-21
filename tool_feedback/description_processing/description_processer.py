import os
import stanza



class DescriptionProcesser:
    def __init__(self, model: str = 'english-extra', version: str = '4.4.0'):
        self.corenlp_folder = self._validate_folder("stanza_corenlp")
        self.model_folder = self._validate_folder("stanza_resources")
        self.model = model
        self.version = version
        self.properties = None
        self.en_nlp = None
        if len(os.listdir(self.corenlp_folder)) == 0:
            print("Installing corenlp...")
            stanza.install_corenlp(dir=self.corenlp_folder)
        else:
            print("Corenlp already installed.")
        if not os.path.isfile(os.path.join(self.model_folder, f"stanford-corenlp-{self.version}-models-{self.model}.jar")):
            print("Downloading English model...")
            stanza.download_corenlp_models(model=self.model, version=self.version, dir=self.model_folder)

    def _validate_folder(self, folder_name: str):
        complete_path = os.path.join("..", "tool_feedback", folder_name)
        if not os.path.exists(complete_path):
            os.makedirs(complete_path)
        return complete_path

    def construct_pipeline(self, language: str="en", properties: str="tokenize,pos,lemma,depparse"):
        # ssplit nativaly true (https://stanfordnlp.github.io/stanza/tokenize.html)
        self.en_nlp = stanza.Pipeline(language, processors=properties,
                                      download_method=stanza.DownloadMethod.REUSE_RESOURCES)

    def annotate_description(self, description: str):
        en_doc = self.en_nlp(description)
        return en_doc



