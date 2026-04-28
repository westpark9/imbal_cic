MAIN   = history
OUTDIR = ../../docs
LATEX  = pdflatex
FLAGS  = -interaction=nonstopmode -halt-on-error

.PHONY: all code3 clean

all:
	$(LATEX) $(FLAGS) $(MAIN).tex
	$(LATEX) $(FLAGS) $(MAIN).tex
	cp $(MAIN).pdf $(OUTDIR)/$(MAIN).pdf
	@echo ">>> Copied to $(OUTDIR)/$(MAIN).pdf"

code3:
	$(LATEX) $(FLAGS) code3.tex
	$(LATEX) $(FLAGS) code3.tex
	cp code3.pdf $(OUTDIR)/code3.pdf
	@echo ">>> Copied to $(OUTDIR)/code3.pdf"

clean:
	rm -f $(MAIN).{aux,log,nav,out,snm,toc,vrb}
	rm -f code3.{aux,log,nav,out,snm,toc,vrb}
