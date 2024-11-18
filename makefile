# Define a rule for converting Jupyter Notebook to HTML
html:
	@jupyter nbconvert --to html $(filter %.ipynb,$(MAKECMDGOALS))

# Prevent make from thinking "input.ipynb" is a file
%:
	@:
