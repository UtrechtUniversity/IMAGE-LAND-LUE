# Documentation Style

The documentation for this package will be written following the [numpy style guide](https://numpydoc.readthedocs.io/en/latest/format.html) in restructured text format, so that it can be rendered in [Sphinx](https://www.sphinx-doc.org/en/master/).
Some key points from the style guide (mostly direct quotes):
* While a rich set of markup is available, we limit ourselves to a very basic subset, in order to provide docstrings that are easy to read on text-only terminals.
* The length of docstring lines should be kept to 75 characters to facilitate reading the docstrings in text terminals.
* The docstring consists of a number of sections separated by headings (except for the deprecation warning; the first section does not require a heading either):
    1. Short Summary, one-line.
    2. Depreciation warning, if applicable.
    3. Extended summary, if required. It may refer to parameters and the function name, but parameter descriptions still belong in the Parameters section.
    4. Parameters, a description of the function arguments, keywords and their respective types.
	    * Formatted as: 
		'param_name : type
					   Description of parameter \`param_name\`'
	    * Enclose variables in single backticks.
	    * The colon must be preceded by a space, or omitted if the type is absent.
	    * If it is not necessary to specify a keyword argument, use `optional`.
	    * If an optional keyword argument has a default value, this can be detailed in the summary, or `optional` can be replaced with `default=default_value`.
    1. Returns, similar to parameters, but the name of each return values is optional; the type is always required.
    2. Yields.
    3. Receives.
    4. Other parameters, to be used in the case where a function has a large number of parameters, some of which are used infrequently.
    5. Raises - optional, detailing which errors get raised and under what conditions.
    6. Warns - similar to raises, but for warnings, rather than errors.
    7. Warnings - optional, to alert the user to possible problems.
    8. See Also - optional, to refer to related code. Routines whose docstrings further explain parameters used by this function are good candidates.
    9. Notes - optional, to provide additional information about the code, possibly including a discussion of the algorithm, which may include equations, in LaTeX format.
	    * Unlike in standard Markdown, math mode is entered by typing ..math:: before the equation, or :math: for an inline statement.
	    * Additional illustrations can be included using: .. image:: filename.
	    * Users viewing the docstring as text need to be taken into account: for this reason, equations should be used judiciously and images, if used, should not be central to the explanation.
    10. References, for references cited in the Notes section.
* Sections should be headed in the style:
```
Parameters
----------
```