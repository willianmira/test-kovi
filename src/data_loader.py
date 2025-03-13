class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        """Carrega os dados do CSV e realiza pré-processamento básico."""
        df = pd.read_csv(self.file_path)
        df['date'] = pd.to_datetime(df['date'])
        return df