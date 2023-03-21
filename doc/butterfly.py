import numpy as np

OX=100
OY=100
def path(x,y, name, color="none"):
    p ="".join(f"L %6.2f,%6.2f "%(x,y) if i else f"M {x},{y} " for i, (x,y) in enumerate(zip(x+OX,y+OY)))
    return f"""<path
       style="fill:{color};stroke:#000000;stroke-width:0.26458332px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1;fill-opacity:0.73869348"
       d="{p}"
       id="{name}"
       inkscape:connector-curvature="0" 
       />"""
def arrow(x0,y0,x1,y1):
    return f"""
    <path
       style="fill:none;stroke:#000000;stroke-width:0.5;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1;stroke-miterlimit:4;stroke-dasharray:none;marker-end:url(#Arrow1Lend)"
       d="M {x0+OX},{y0+OY} L {x1+OX},{y1+OY}"
       />
    """
def arrows(x0,y0,x1,y1):
    text = "<g>\n"
    for i in range(len(x0)):
        text+=arrow(x0[i],y0[i],x1[i],y1[i])
    text += "</g>\n"
    return text

a=5.0
b=1
N=100
R=100
epsilon = 1e-5
alpha=0.99

angle = np.linspace(0.0, 2*np.pi, N)

x = R*np.sin(angle)
y = R*np.cos(angle)


def do_step(x,y, alpha=1.0, epsilon=1e-5):
    gx = a*x
    gy = b*y
    ng = np.sqrt(gx*gx + gy*gy)

    xx=x+epsilon*gx/ng
    yy=y+epsilon*gy/ng
    ggx = a*xx
    ggy = b*yy

    dgx = ggx-gx
    dgy = ggy-gy
    dg = np.sqrt(dgx*dgx + dgy*dgy)

    step = alpha*epsilon/dg

    x1 = x - step*gx
    y1 = y - step*gy
    return x1,y1

x1,y1 = do_step(x,y,alpha=alpha)
x2,y2 = do_step(x1,y1,alpha=alpha)

#x1 = x - 20*gx/ng
#y1 = y - 20*gy/ng




with open("out.svg","w") as f:
    f.write(f"""
<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->

<svg
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
   width="210mm"
   height="297mm"
   viewBox="0 0 210 297"
   version="1.1"
   id="svg8"
   inkscape:version="0.92.3 (2405546, 2018-03-11)"
   sodipodi:docname="drawing.svg">
  <defs
     id="defs2">
    <marker
       inkscape:stockid="Arrow1Lend"
       orient="auto"
       refY="0.0"
       refX="0.0"
       id="Arrow1Lend"
       style="overflow:visible;"
       inkscape:isstock="true">
      <path
         id="path856"
         d="M 0.0,0.0 L 5.0,-5.0 L -12.5,0.0 L 5.0,5.0 L 0.0,0.0 z "
         style="fill-rule:evenodd;stroke:#000000;stroke-width:1pt;stroke-opacity:1;fill:#000000;fill-opacity:1"
         transform="scale(0.8) rotate(180) translate(12.5,0)" />
    </marker>
  </defs>
  <sodipodi:namedview
     id="base"
     pagecolor="#ffffff"
     bordercolor="#666666"
     borderopacity="1.0"
     inkscape:pageopacity="0.0"
     inkscape:pageshadow="2"
     inkscape:zoom="0.7"
     inkscape:cx="-128.57143"
     inkscape:cy="560"
     inkscape:document-units="mm"
     inkscape:current-layer="layer1"
     showgrid="false"
     inkscape:window-width="1920"
     inkscape:window-height="1026"
     inkscape:window-x="0"
     inkscape:window-y="0"
     inkscape:window-maximized="1" />
  <metadata
     id="metadata5">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title></dc:title>
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     inkscape:label="Layer 1"
     inkscape:groupmode="layer"
     id="layer1">
    {path(x,y,'p0','#bbccdd')}
    {path(x1,y1,'p1','#aabbcc')}
    {path(x2,y2,'p2','#ccddaa')}
    {arrows(x,y,x1,y1)}
    <!--
    {arrows(x1,y1,x2,y2)}
    -->
  </g>
</svg>
""")