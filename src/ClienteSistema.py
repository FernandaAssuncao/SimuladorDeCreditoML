class Cliente:
    def __init__(self, idade:int, salario:float, nome_limpo:int):
        self.idade = idade
        self.salario = salario
        self.nome_limpo = nome_limpo


class PropostaEmprestimo(Cliente):
    def __init__(self, idade:int, salario:float, nome_limpo:int, valor:float):
        super().__init__(idade, salario, nome_limpo)
        self.valor_solicitado = valor
        self._status = 'pendente'

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, status:str):
        if status == 'aprovado' or status == 'reprovado':
            self._status = status
        else:
            raise ValueError("Status deve ser 'aprovado' ou 'reprovado'.")
