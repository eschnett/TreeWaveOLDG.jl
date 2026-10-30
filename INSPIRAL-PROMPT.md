the paper https://arxiv.org/abs/2606.06487 describes a numerical
algorithm for handling small particles in a numerical simulation of a
scalar field. its key features are:

- the particle has a small but finite size \ell, comparable to the
  grid spacing h. keeping it finite ensures consistency without having
  to handle singularities.
- the field is split into a particle field and remainder field.
- the particle field describes a non-accelerating particle and is
  handled analytically. it changes with the position and velocity of
  the particle.
- the remainder field is discretized numerically.
- an accelerating particle has a backreaction, i.e. it adds to the RHS
  of the remainder field.

note also:
- the case \ell >> h is not very interesting because a straightforward
  numerical method can resolve the particle; it is not "small" any
  more.
- the case \ell << h is difficult. it leads to a point particle which
  is non-trivial to handle.
- we are interested in the the case \ell \approx h. a convergence test
  would keep \ell/h constant.

design a numerical method, based on the implemented OLDG scheme, to
simulate two inspiralling scalar a particles, starting on a circular
orbit. this method needs to handle the backreaction from the particle
onto the remainder field, as well as determine the forces onto the
particle from the remainder field.

use 3 spatial dimensions.

ignore outer boundary effects, i.e. assume that the outer boundary
causally is disconnected from the interesting region in the interior.

only develop the numerical method. do not implement anything. present
the method in a markdown file. use a new git worktree named
`TreeWaveOLDG-inspiral-Opus5.5-xhigh` and a feature branch. do not
read any other directories named `TreeWaveOLDG-inspiral*`.

ask me to clarify if these instructions are unclear or lacking.
