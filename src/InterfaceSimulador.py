from .ClienteSistema import PropostaEmprestimo
from .IASimuladorDeImprestimo import IAFinanceira
import customtkinter as ctk
from .GerenciadorDeDados import GerenciarDados


class InterfaceSimuladorImprestimo(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.ia = IAFinanceira()

        self.title('Simulador de Imprestimo')
        self.geometry('600x650')
        self._set_appearance_mode('dark')
        self.columnconfigure(0, weight=1)

        self.titulo = ctk.CTkLabel(self, text='Simulador de Empréstimo IA', text_color='#00BFFF', fg_color='#1A1A1A',
                                   font=('Century Gothic', 25, 'bold'), height=60, corner_radius=15)
        self.titulo.grid(row=0, column=0, columnspan=4, padx=20, pady=20, sticky='nsew')

        self.mensagem = ctk.CTkLabel(self, text='Digite abaixo as informações solicitadas: IDADE',
                                     text_color='#1E90FF', font=('Arial', 15, 'bold'))
        self.mensagem.grid(row=1, column=0, columnspan=2, padx=20, pady=20, sticky='nsew')

        self.campo_idade = ctk.CTkEntry(self, border_color='#00BFFF', border_width=2,
                                        corner_radius=50)
        self.campo_idade.grid(row=1, column=2, columnspan=1, padx=20, pady=20, sticky='nsew')
        self.campo_idade.bind("<Return>", lambda event: self.campo_salario.focus_set())

        self.mensagem2 = ctk.CTkLabel(self, text='SALARIO: ', text_color='#1E90FF',
                                      font=('Arial', 15, 'bold'))
        self.mensagem2.grid(row=2, column=0, columnspan=1, padx=20, pady=20, sticky='nsew')

        self.campo_salario = ctk.CTkEntry(self, border_color='#00BFFF', border_width=2,
                                          corner_radius=50)
        self.campo_salario.grid(row=2, column=2, padx=20, pady=20, sticky='nsew')
        self.campo_salario.bind("<Return>", lambda event: self.campo_valor.focus_set())

        self.mensagem3 = ctk.CTkLabel(self, text='VALOR DO EMPRÉSTIMO: ', text_color='#1E90FF',
                                      font=('Arial', 15, 'bold'))
        self.mensagem3.grid(row=3, column=0, columnspan=1, padx=20, pady=20, sticky='nsew')

        self.campo_valor = ctk.CTkEntry(self, border_color='#00BFFF', border_width=2,
                                        corner_radius=50)
        self.campo_valor.grid(row=3, column=2, columnspan=1, padx=20, pady=20, sticky='nsew')
        self.campo_valor.bind("<Return>", lambda event: self.gerar_resposta())


        self.mensagem4 = ctk.CTkLabel(self, text='SEU NOME ESTÁ LIMPO?\n [SIM] ligado e [NÃO] desligado ',
                                      text_color='#1E90FF', font=('Arial', 15, 'bold'))
        self.mensagem4.grid(row=4, column=0, columnspan=1, padx=20, pady=20, sticky='nsew')

        self.switch_nome_limpo = ctk.CTkSwitch(
            self,
            text="Nome Limpo",
            font=('Arial', 14, 'bold'),
            progress_color='#00BFFF'  # Cor quando estiver ligado
        )
        self.switch_nome_limpo.grid(row=4, column=2, columnspan=2, padx=20, pady=20, sticky='nsew')

        self.resposta = ctk.CTkLabel(self, text='', text_color='#1E90FF', font=('Arial', 15, 'bold'))
        self.resposta.grid(row=5, column=0, columnspan=2, padx=20, pady=20, sticky='nsew')

        self.botao = ctk.CTkButton(self, command=self.gerar_resposta, text='ANALISAR',
                                   text_color='white', fg_color='#00BFFF',
                                   border_color='#00FFFF', corner_radius=15, border_width=2,
                                   font=('Century Gothic', 19, 'bold'), hover_color='#008B8B')
        self.botao.grid(row=5, column=2, columnspan=2, padx=20, pady=20, sticky='nsew')

        self.frame = ctk.CTkFrame(self, border_color='#00BFFF', border_width=1)
        self.frame.grid(row=6, column=0, columnspan=4, padx=20, pady=20, sticky='nsew')

        self.mensagem_frame = ctk.CTkLabel(self.frame, text='Historico de\n Pedidos',
                                           text_color='#1E90FF', font=('Arial', 16, 'bold'))
        self.mensagem_frame.grid(row=0, column=0, columnspan=1, padx=20, pady=20, sticky='nsew')

        self.conteudo_frame = ctk.CTkTextbox(
            self.frame,
            height=100,  # Altura fixa com rolagem interna
            text_color="#1E90FF",
            font=("Arial", 12, "bold"),
            border_color="#00BFFF",
            border_width=1,
            corner_radius=10,
        )
        self.conteudo_frame.grid(row=0, column=3, columnspan=2, padx=20, pady=20, sticky='nsew')
        self.frame.columnconfigure(0, weight=1)

        self.colocar_conteudo()


    def gerar_resposta(self):
        try:
            idade = int(self.campo_idade.get())
            salario = float(self.campo_salario.get())
            valor = float(self.campo_valor.get())
            nome_limpo = int(self.switch_nome_limpo.get())
            if nome_limpo == 1:
                historico_inadimplencia = 0
            else:
                historico_inadimplencia = 1
            renda_anual = salario * 12
            percentual_renda = valor / renda_anual
            print(percentual_renda)
            cliente = PropostaEmprestimo(idade, salario, nome_limpo, valor)
            previsao = self.ia.prever(idade, salario, valor, percentual_renda, historico_inadimplencia)
            if previsao:
                probabilidade = self.ia.calcular_probabilidade_aprovado(idade, salario, valor, percentual_renda, historico_inadimplencia)
                cliente.status = 'aprovado'
                cor = '#7FFFD4'
                texto = f'Parabéns, o Empréstimo no valor de\n R${cliente.valor_solicitado:.2f} foi {cliente.status}.\n Probabilidade de aprovação estimada {probabilidade * 100:.2f}%'
            else:
                probabilidade = self.ia.calcular_probabilidade_reprovado(idade, salario, valor, percentual_renda, historico_inadimplencia)
                cliente.status = 'reprovado'
                cor = '#9B111E'
                texto = f'Sinto muito, o Empréstimo no valor de\n R${cliente.valor_solicitado:.2f} foi {cliente.status}.\n Probabilidade de reprovação estimada {probabilidade* 100:.2f}%'
            self.resposta.configure(text=texto, text_color=cor)
            self.campo_idade.configure(border_color=cor)
            self.campo_salario.configure(border_color=cor)
            self.campo_valor.configure(border_color=cor)
            g = GerenciarDados()
            g.salvar_novo_dado(cliente.idade, cliente.salario, cliente.valor_solicitado,
                               cliente.nome_limpo, cliente.status)
            self.colocar_conteudo()
        except ValueError:
            cor = '#FF3B30'
            self.campo_idade.configure(border_color=cor)
            self.campo_salario.configure(border_color=cor)
            self.campo_valor.configure(border_color=cor)
            self.resposta.configure(text='ERRO, por favor digite\n O que se pede corretamente!',
                                    text_color=cor)

    def colocar_conteudo(self):
        g = GerenciarDados()
        conteudo = g.gerar_conteudo_para_historico()
        self.conteudo_frame.configure(
            state="normal"
        )  # Permite alterar o texto
        self.conteudo_frame.delete("0.0", "end")  # Limpa o conteúdo anterior
        self.conteudo_frame.insert("0.0", conteudo)  # Insere o novo histórico
        self.conteudo_frame.see("end")  # Rola automaticamente para o final
        self.conteudo_frame.configure(
            state="disabled"
        )
