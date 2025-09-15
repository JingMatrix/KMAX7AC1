%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fichier de definition de la classe de documents
% exercices
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Cette classe s'utilise en commencant le document par
%\documentclass{exercices}
% et si on veut afficher les solutions par
%\documentclass[solutions]{exercices}

% Ensuite, on ecrit un exercice par
% \begin{exercice}\end{exercice}
% et ses solutions par
% \begin{solution}\end{solution}

% Les solutions ne sont affichees que si l'option [solutions] est donnee en en tete
% de la commande \documentclass

\LoadClass[a4paper,12pt]{article}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Variable qui dit si on souhaite
% afficher les solutions (1) ou non (0)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\affichesolution}{0}
\DeclareOption{solutions}
   {\renewcommand{\affichesolution}{1}}
\ProcessOptions

%%%%%%%%%%%%%%%%%%%%%%%%%%
% Choix des packages latex
%%%%%%%%%%%%%%%%%%%%%%%%%%
\RequirePackage{a4}
\RequirePackage[french]{babel}
\RequirePackage{array}
\RequirePackage{amsfonts,amssymb,amsmath,amscd,amsthm}
\RequirePackage{mathrsfs}
\RequirePackage{bbm}
\RequirePackage{ifthen}
\RequirePackage{comment}
\RequirePackage{tikz}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Definition des commandes pour
% les caracteres mathematiques
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\R}{\mathbb{R}}
\newcommand{\C}{\mathbb{C}}
\newcommand{\Z}{\mathbb{Z}}
\newcommand{\Q}{\mathbb{Q}}
\newcommand{\N}{\mathbb{N}}
\newcommand{\F}{\mathbb{F}}
\newcommand{\A}{\mathbb{A}}
\newcommand{\1}{\mathbbm{1}}
\newcommand{\ind}[1]{{\1_{\left\{#1\right\}}}}
\newcommand{\dd}{\mathrm{d}}
\newcommand{\norme}[1]{{\left\Vert#1\right\Vert}}
\newcommand{\E}{\mathbb{E}}
\renewcommand{\P}{\mathbb{P}}
\newcommand{\floor}[1]{{\left\lfloor#1\right\rfloor}}
\renewcommand{\bar}[1]{\overline{#1}}

\renewcommand{\Re}{\mathrm{Re}\,}
\renewcommand{\Im}{\mathrm{Im}\,}
\newcommand{\Log}{\mathrm{Log}\,}
\newcommand{\dlog}{\mathrm{dlog}\,}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Definition des commandes pour
% les caracteres arrondis
% Par convention, on ajoute c
% a la suite du caractere
% souhaite
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\Ac}{\mathcal{A}}
\newcommand{\Bc}{\mathcal{B}}
\newcommand{\Cc}{\mathcal{C}}
\newcommand{\Dc}{\mathcal{D}}
\newcommand{\Ec}{\mathcal{E}}
\newcommand{\Gc}{\mathcal{G}}
\newcommand{\Ic}{\mathcal{I}}
\newcommand{\Lc}{\mathcal{L}}
\newcommand{\Mc}{\mathcal{M}}
\newcommand{\Nc}{\mathcal{N}}
\newcommand{\Oc}{\mathcal{O}}
\newcommand{\Pc}{\mathcal{P}}
\newcommand{\Sc}{\mathcal{S}}
\newcommand{\Tc}{\mathcal{T}}

\newcommand{\Isc}{\mathscr{I}}
\newcommand{\Csc}{\mathscr{C}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Commandes mathematiques moins standards
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\Rbp}{\bar{\R}_+}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Correction des problemes d'accents
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\def \é{\'e}
\def \è{\`{e}}
\def \ù{\`{u}}
\def \î{\^{\i}}
\def \à{\`{a}}
\def \ç{\c{c}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Mise en page a la main de la feuille d'exercices
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\parindent=0pt
\textwidth 17.0cm
\textheight 24.0cm
\oddsidemargin 0.4cm
\evensidemargin -0.4cm
\voffset -2cm
\hoffset -1cm

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Redefinition des labels pour les enumerations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%\renewcommand{\labelenumi}{\alph{enumi})\,}
%\renewcommand{\labelenumii}{\roman{enumii})\,}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Definition d'une commande pour afficher l'annee suivante
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcounter{nextyear}\setcounter{nextyear}{\number\year}
\addtocounter{nextyear}{1}\newcommand{\nextyear}{\arabic{nextyear}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Definition des en-tetes et compteurs pour les exercices
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcounter{exercice}
\setcounter{exercice}{0}
\newenvironment {exercice}
		{\refstepcounter{exercice}
		 {\bf Exercice \arabic{exercice}. }}
		{\bigskip}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Definition des en-tetes et compteurs pour les solutions
% On ne les affiche que si la commande \affichesolution
% ne vaut pas 0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcounter{solution}
\setcounter{solution}{0}

% On teste la variable affichesolution
\ifthenelse{\equal{\affichesolution}{0}}
% Si elle est egale a 0, on voit l'environnement solution
% comme un commentaire a exclure
	   {\excludecomment{solution}}
% Si la variable n'est pas egale a 0, on definit l'environnement
% solution pour qu'il s'affiche comme on le souhaite
	   {\newenvironment{solution}
		{\setcounter{solution}{\value{exercice}}
		\emph{Solution de l'exercice} \arabic{solution}.}
		{\bigskip}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Definition des en-tetes et compteurs pour les easter eggs
% On ne les affiche que si la commande \affichesolution
% ne vaut pas 0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcounter{easteregg}
\setcounter{easteregg}{0}

% On teste la variable affichesolution
\ifthenelse{\equal{\affichesolution}{0}}
% Si elle est egale a 0, on voit l'environnement solution
% comme un commentaire a exclure
	   {\excludecomment{easteregg}}
% Si la variable n'est pas egale a 0, on definit l'environnement
% solution pour qu'il s'affiche comme on le souhaite
	   {\newenvironment{easteregg}
		{\setcounter{easteregg}{\value{exercice}}
		\emph{Exercice} \Alph{easteregg}.}
		{\bigskip}}


% Styles de théorèmes pouvant être utile
\theoremstyle{plain}
\newtheorem{theoreme}{Théorème}

%% Ajouté par Adrien : la commande \important qui met un i dans la marge. A faire évoluer.
\newcommand{\important}{
\setlength{\unitlength}{1mm}%
\begin{picture}(0,0)%
\put(-30,0){\makebox[0pt]{$\star$}}%
\end{picture}%
}
