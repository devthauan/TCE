def salvar(y_test,y_predito,micro, macro, f1_individual,texto):
    f = open('Resultados.txt','a+')
    f.write("O f1Score micro do "+texto +" é: "+str(micro)+'\n')
    f.write("O f1Score macro do "+texto +" é: "+str(macro)+'\n')
    f.write('\n')
    f.write("f1 individual para cada natureza"+'\n')
    for score in f1_individual:
        f.write(str(score)+'\n')
    f.write('\n')
    f.flush()
    f.close()