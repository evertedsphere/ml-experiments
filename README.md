# ML in Emacs with jupyter-org / Org mode

See [makemore](./makemore/) for where I'm at right now.

## Emacs setup
I use Doom Emacs, where the bulk of the setup is done once you've
put `(org +jupyter)` and `python` in your modules. 

A bit of coaxing is required to actually get Jupyter to work, though:
```emacs-lisp
;; Needed on Doom Emacs
(advice-remove #'org-babel-do-load-languages #'ignore)

;; The order matters here!
(org-babel-do-load-languages
 'org-babel-load-languages
 '((emacs-lisp . t)
   (shell . t)
   (python . t)
   (jupyter . t)))
```
  
This will let you remove the `:session` boilerplate, among other things:
```emacs-lisp
(setq org-babel-default-header-args:jupyter-python 
  '((:kernel . "ipython_python-env")
    (:display . "text/plain")
    (:session . "emacs-python-session")
    (:exports . "both")
    (:async . "yes")))
```

Finally, a quick QoL improvement:
``` emacs-lisp
;; C-c C-,
(add-to-list 'org-structure-template-alist '("py" . "src jupyter-python"))
```

### Acknowledgements
Nix setup stolen from [here](https://github.com/collinarnett/stable-diffusion-nix).
