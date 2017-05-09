for line in open("dataset_movies.txt","r"):
	line = line.replace("\r","").replace("\n","")
	tok = line.split("\t")
	print "{}".format(tok[0])
