MAIN   = history
OUTDIR = ../../docs
LATEX  = pdflatex
XELATEX = xelatex
FLAGS  = -interaction=nonstopmode -halt-on-error

.PHONY: all code3 ood_tta ood_tta_figure clean

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

ood_tta:
	$(XELATEX) $(FLAGS) ood_tta_draft.tex
	$(XELATEX) $(FLAGS) ood_tta_draft.tex
	@echo ">>> Built ood_tta_draft.pdf"

ood_tta_figure:
	$(XELATEX) $(FLAGS) ood_tta_main_figure.tex
	@echo ">>> Built ood_tta_main_figure.pdf"

clean:
	rm -f $(MAIN).{aux,log,nav,out,snm,toc,vrb}
	rm -f code3.{aux,log,nav,out,snm,toc,vrb}
	rm -f ood_tta_draft.{aux,log,nav,out,snm,toc,vrb}
	rm -f ood_tta_main_figure.{aux,log}
