\documentclass[11pt]{article}
\begin{document}
The difficulty in implementing the SINDY fit is that the basis functions are matrices and powers of matrices. 
There are at wo possible approaches: 

- Expand the basis matrix functions into their individual components. Since the basis functions are 3x3 and are symmetric matrices. 

- Somehow treat matrices as matrices. Come up with a algorithm that operates on matrices. This would require deeloping  new algorithm. Perhaps I should understand how RUDE works. 

The first would be to express the basis functions assuming that \sigma_{12} and \sigma{23} are zero.

#-----------------------------------------------------------------------------------------------
In Sachin's project, are we seeking an analytical expression of the unknown F(...) that is parametrized by 
\lambda, G, etc? Or are we fitting the term for each set of parameter and initial condition values (which is easier, but still hard.) 

\end{document}
