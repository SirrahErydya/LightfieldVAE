# LightfieldVAE

Multi-camera light fields impress with their precision and high-quality results. Nevertheless, they undergo neglection due to their high hardware expenses and cumbersome setup. By reducing the amount of cameras and generating the missing viewpoint captures instead, the effort can be narrowed. This work introduces a method to interpolate between scene images along the horizontal and vertical axis in order to augment the images along the diagonal axis.

By training a variational autoencoder on the image stacks of cross-shaped light field captures, we were able to generate the decreasing diagonals for distinct input cross-stacks.
