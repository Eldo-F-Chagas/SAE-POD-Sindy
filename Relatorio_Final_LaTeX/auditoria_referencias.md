# Auditoria das referências do pré-projeto

Data da auditoria: 29 de setembro de 2026.

## Critério adotado

As referências do pré-projeto e do relatório semestral foram conferidas contra registros oficiais (Crossref, arXiv, NASA NTRS, OpenAlex e documentação do OpenFOAM). Uma afirmação foi mantida no relatório somente quando seu conteúdo estava explícito no texto integral, no resumo oficial ou, para afirmações estritamente temáticas, no próprio título. A mera presença de uma palavra isolada não foi tratada como comprovação de uma conclusão científica.

Níveis de verificação:

- **texto integral**: conteúdo do documento consultado;
- **resumo oficial**: escopo e conclusões limitados ao resumo da editora/repositório;
- **metadados/título**: usados apenas para autoria, título, veículo, ano, DOI e descrição temática estrita;
- **livro**: usado somente para conceitos gerais compatíveis com o título/escopo e dados editoriais confirmados; não foram atribuídas frases ou conclusões específicas sem página verificável.

## Fontes incorporadas e uso permitido no relatório

| Referência | Verificação | Conteúdo efetivamente sustentado e usado |
|---|---|---|
| Brunton e Kutz (2019) | metadados oficiais da Cambridge/Crossref | Obra sobre ciência e engenharia orientadas por dados, aprendizado de máquina, sistemas dinâmicos e controle. Citada apenas como fundamento geral. |
| Brunton, Proctor e Kutz (2016) | resumo oficial e metadados Crossref | SINDy identifica equações governantes por regressão esparsa e busca modelos parcimoniosos. |
| Corbetta (2020) | texto integral e registro NASA NTRS 20200001544 | Regressão esparsa para extrair equações governantes e construir modelos substitutos fisicamente informados; o texto também discute interpretabilidade e quantidade de dados. |
| Díaz, Barreiro e Rubido (2023) | resumo oficial da editora/Crossref | Modelos bidimensionais da oscilação Madden--Julian inferidos com SINDy e análise de anos El Niño/La Niña. |
| Dixit et al. (2025) | resumo oficial do arXiv | Ambientes substitutos baseados em SINDy para aprendizado por reforço nos problemas Mountain Car e Lunar Lander. |
| Kaiser, Kutz e Brunton (2018) | resumo oficial/Crossref | Combinação SINDy--MPC e avaliação no limite de poucos dados, inclusive com dados ruidosos. |
| Kaptanoglu et al. (2022) | resumo oficial do arXiv e publicação JOSS/Crossref | PySINDy; bibliotecas para sistemas atuados, PDEs e equações implícitas; formulação integral, ensemble, restrições e diversos otimizadores. |
| Lin, Xiao e Fang (2024) | texto/resumo do arXiv 2401.05449 | Autoencoder para variedade não linear, POD para estabilizar o espaço latente e regressão esparsa para descobrir dinâmica reduzida de escoamentos. |
| Méndez et al. (2023) | metadados oficiais do livro e capítulos registrados pela Cambridge/Crossref | Referência geral sobre mecânica dos fluidos orientada por dados, redução dimensional e identificação de sistemas. |
| Naozuka et al. (2022) | título e metadados oficiais Crossref/OpenAlex | O artigo propõe o framework SINDy-SA e relaciona identificação não linear com análise de sensibilidade. Não foram atribuídos resultados numéricos ao artigo. |
| OpenFOAM Foundation (2025) | documentação oficial da versão 13 | Organização e uso geral do OpenFOAM. Os parâmetros particulares do relatório foram conferidos nos arquivos dos casos, e não inferidos do manual. |
| Pritchard e Mitchell (2011) | dados editoriais do livro | Fundamentos gerais de mecânica dos fluidos e leis de conservação; nenhuma conclusão específica foi atribuída sem indicação de página. |
| Regis, John Mathuram e Padiyarajan (2023) | resumo indexado e metadados Crossref/OpenAlex | Aplicação de SINDy para previsão de temperatura e umidade relativa. |
| Rudy et al. (2017) | resumo oficial e metadados Crossref | Regressão esparsa para descoberta de equações diferenciais parciais em sistemas espaço-temporais. |
| Tu, Yeoh e Liu (2008) | dados editoriais do livro e metadados dos capítulos | Fundamentos e prática de CFD. Citado apenas para conservação, discretização e descrição geral de campos computacionais. |
| Zhong, Jiang e Zhang (2024) | título e metadados oficiais Crossref/OpenAlex | TC--SINDy aplicado a um modelo determinístico de trajetória e intensidade de ciclones tropicais. Não foram atribuídas métricas não verificadas. |

## Correções bibliográficas feitas

1. **Méndez et al.**: o DOI `10.1017/9781009310819`, presente no relatório semestral, não corresponde ao livro *Data-Driven Fluid Mechanics*; ele aponta para *Giftedness in Childhood*. O DOI correto inserido no relatório é **10.1017/9781108896214**.
2. **Regis et al.**: o evento foi corrigido de “IEEE ICAICA” para **2023 12th International Conference on Advanced Computing (ICoAC)**, e o DOI correto é **10.1109/ICoAC59537.2023.10249738**.
3. **Kaptanoglu et al.**: o pré-projeto citava o preprint de 2021. A bibliografia final usa a publicação revisada do *Journal of Open Source Software*, de 2022, DOI **10.21105/joss.03994**.
4. **Lin, Xiao e Fang**: o título completo foi registrado como *Data discovery of low dimensional fluid dynamics of turbulent flows*.
5. **Corbetta**: o título completo foi corrigido para *Application of Sparse Identification of Nonlinear Dynamics for Physics-Informed Learning* e associado ao registro NASA NTRS 20200001544.
6. **Díaz et al.** e **Zhong et al.**: foram acrescentados número de artigo e DOI confirmados.
7. **OpenFOAM**: a entrada genérica de 2021 foi atualizada para a documentação da versão 13 efetivamente relacionada ao ambiente analisado.

## Limites deliberados

- O relatório não afirma que as referências ambientais cubram dispersão de poluentes, formação de nuvens ou circulação oceânica. Esses temas aparecem como possibilidades do projeto, pois as fontes auditadas citadas no texto tratam especificamente de oscilação Madden--Julian, previsão de temperatura/umidade e ciclones tropicais.
- O trabalho de Zhong et al. e o de Naozuka et al. foram usados somente no nível explicitamente comprovável por título e metadados, pois o texto integral não foi necessário nem usado para sustentar métricas ou conclusões específicas.
- Os livros foram usados como fundamento geral. Não foram fabricadas citações textuais, números de página ou resultados experimentais.
- Resultados numéricos de POD, SAE e SINDy no relatório têm como fonte os próprios artefatos computacionais do projeto; por isso, não recebem referência externa como se viessem da literatura.
