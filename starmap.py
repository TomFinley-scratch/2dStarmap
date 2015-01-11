# Sci-fi games often present a 2D star map to players, despite the obvious
# incompatibility with the reality of 3D space. Any such presentation must
# wrong, but I was slightly curious, if one were to build a map of the
# stars around the sun, what would be the "least bad" presentation of such
# a thing as that? My thought was, a locally linear embedding might be a
# good choice for such data.

# This was a very small "evening" project I threw together, early November
# 2010. I never worked on it beyond that, but I found it sufficiently
# interesting as to be worth some small effort of preservation.

import gzip
import heapq
import itertools
import collections

import numpy
import mdp
import scipy.optimize
import lle_nodes2

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pylab

class HIPID(int):
    iid2name = {0:"Sun", 13847:"Acamar", 57939:"Groombridge 1830", 7588:"Achernar",
                68702:"Hadar", 60718:"Acrux", 9884:"Hamal", 33579:"Adhara", 72105:"Izar",
                68702:"Agena", 24186:"Kapteyn's star", 95947:"Albireo",
                90185:"Kaus Australis", 65477:"Alcor", 72607:"Kocab", 17702:"Alcyone",
                110893:"Kruger 60", 21421:"Aldebaran", 36208:"Luyten's star",
                105199:"Alderamin", 113963:"Markab", 1067:"Algenib", 59774:"Megrez",
                50583:"Algieba", 14135:"Menkar", 14576:"Algol", 53910:"Merak",
                31681:"Alhena", 25930:"Mintaka", 62956:"Alioth", 10826:"Mira",
                67301:"Alkaid", 5447:"Mirach", 9640:"Almaak", 15863:"Mirphak",
                109268:"Alnair", 65378:"Mizar", 25428:"Alnath", 25606:"Nihal",
                26311:"Alnilam", 92855:"Nunki", 26727:"Alnitak", 58001:"Phad",
                46390:"Alphard", 17851:"Pleione", 76267:"Alphekka", 11767:"Polaris",
                677:"Alpheratz", 37826:"Pollux", 98036:"Alshain", 37279:"Procyon",
                97649:"Altair", 70890:"Proxima", 2081:"Ankaa", 84345:"Rasalgethi",
                80763:"Antares", 86032:"Rasalhague", 69673:"Arcturus",
                30089:"Red Rectangle", 25985:"Arneb", 49669:"Regulus",
                112247:"Babcock's star", 24436:"Rigel", 87937:"Barnard's star",
                71683:"Rigil Kent", 25336:"Bellatrix", 109074:"Sadalmelik",
                27989:"Betelgeuse", 27366:"Saiph", 96295:"Campbell's star",
                113881:"Scheat", 30438:"Canopus", 85927:"Shaula", 24608:"Capella",
                3179:"Shedir", 746:"Caph", 92420:"Sheliak", 36850:"Castor",
                32349:"Sirius", 63125:"Cor Caroli", 65474:"Spica", 98298:"Cyg X-1",
                97278:"Tarazed", 102098:"Deneb", 68756:"Thuban", 57632:"Denebola",
                77070:"Unukalhai", 3419:"Diphda", 3829:"Van Maanen 2", 54061:"Dubhe",
                91262:"Vega", 107315:"Enif", 63608:"Vindemiatrix", 87833:"Etamin",
                18543:"Zaurak", 113368:"Fomalhaut", 60936:"3C 273"}
    def __str__(self):
        return self.iid2name.get(self, "")

def hipdata(closest = None):
    # Read in the Hipparcos data.
    hipfile = gzip.open('hip_main.dat.gz', 'rb')

    no_parallax = 0                   # Count the non-data queries.
    hip_ids = [HIPID(0)]              # Pretend the sun has HIP ID 0.
    coordinates = [[0.0, 0.0, 0.0]]   # Initialize with the sun at origin.

    cc = collections.defaultdict(int)

    print 'Reading Hipparcos data...'

    for linenum, line in enumerate(hipfile):
        #if linenum==1000: break
        hip_id = HIPID(line[8:14])    # Hipparcos ID number
        try:
            parallax = float(line[79:86]) # Parallax is in mas
        except ValueError:
            # Some parallax entries are blank.
            no_parallax += 1
            continue
        if parallax == 0.0:
            no_parallax += 1
            continue
        parsec   = 1000.0 / parallax  # Calculate distance
        radeg    = float(line[51:63]) # RA in degrees
        dedeg    = float(line[64:76]) # Declination in degrees
        rarad, derad = numpy.deg2rad(radeg), numpy.deg2rad(dedeg)

        # Convert into 3D coordinates (with the sun at origin)
        # north pole has X=0, Y=0, Z=1
        # equator at RA 0 has X=1, Y=0, Z=1
        z = numpy.sin(derad)
        factor = (1-z*z)**.5
        x, y = numpy.cos(rarad)*factor, numpy.sin(rarad)*factor

        hip_ids.append(hip_id)
        coordinates.append([x*parsec, y*parsec, z*parsec])

    print 'processed %d lines, %d skipped' % (linenum, no_parallax)
    # Get however many are closest.
    if closest:
        psq_id_coord = heapq.nsmallest(closest, ((x*x+y*y+z*z, iid, (x,y,z)) for iid,(x,y,z) in itertools.izip(hip_ids, coordinates)))
        hip_ids = [iid for psq,iid,coord in psq_id_coord]
        coordinates = [coord for psq,iid,coord in psq_id_coord]
    # Convert this into a numpy array.
    coordinates = numpy.array(coordinates)
    hipfile.close()

    return hip_ids, coordinates

def closest100data():
    infile = open('closest100.txt', 'rb')

    identifiers = ['Sun']
    coordinates = [[0.0, 0.0, 0.0]]
    
    for line in infile:
        # Only pay attention to systems.
        idnum = line[:3].strip()
        if not idnum: continue
        idnum = int(idnum)
        # Get the names.
        name = line[152:].strip()
        if not name or 'et al.' in name: name = ' '.join(line[5:21].split())
        identifiers.append(name)
        # Get the parallax.
        parallax = float(line[73:80])
        parsec   = 1.0 / parallax
        # Get the RA.
        radeg = float(line[32:34])*15 + float(line[35:37])/4.0 + float(
            line[38:42])/240.0
        rarad = numpy.deg2rad(radeg)
        # Get the dec.
        dedeg = float(line[44:46]) + float(line[47:49])/60.0 + float(
            '0'+line[50:52])/3600.0
        derad = numpy.deg2rad(dedeg)
        
        # Convert into 3D coordinates (with the sun at origin)
        # north pole has X=0, Y=0, Z=1
        # equator at RA 0 has X=1, Y=0, Z=1
        z = numpy.sin(derad)
        factor = (1-z*z)**.5
        x, y = numpy.cos(rarad)*factor, numpy.sin(rarad)*factor

        coordinates.append([x*parsec, y*parsec, z*parsec])

    coordinates = numpy.array(coordinates)

    return identifiers, coordinates

def hygdata(named_only=False, exp=1):
    infile = gzip.open('hygxyz.csv.gz', 'rb')
    identifiers = []
    coordinates = []
    infile.next() # skip first line
    for line in infile:
        tokens = line.split(',')
        if len(tokens)!=23: continue
        name = tokens[6]
        if named_only and not name: continue
        coords = [float(tokens[i]) for i in (17,18,19)]
        distance = float(tokens[9])
        if distance < 2.5: print distance, line
        nn = numpy.linalg.norm(coords)
        dd = nn**exp
        coords = [(x*dd/nn if nn else x) for x in coords]

        identifiers.append(name)
        coordinates.append(coords)

    coordinates = numpy.array(coordinates)
    return identifiers, coordinates

#ids, coords = hipdata(1000)
ids, coords = closest100data()
#ids, coords = hygdata(True, .4)

print 'acquired %d stars' % len(ids)

#for iid, coord in zip(ids, coords): print iid, coord

print 'running projection'
LL = mdp.nodes.LLENode
LL = lle_nodes2.LLENode
imp = [0]+[1]*(len(ids)-1)
projector = LL(8, output_dim=2, verbose=True, importances=imp)
projected = projector(coords)
print 'done!'

# Get "error" w.r.t. a given star.  First compute the proper slope.
central = 0
true_dists = [numpy.linalg.norm(coords[central]-c) for c in coords]
proj_dists = [numpy.linalg.norm(projected[central]-p) for p in projected]
average_slope = numpy.mean([
    t/p for (t,p) in itertools.izip(true_dists, proj_dists) if p])
def slope_error(invslope):
    return sum((1-(t/p*invslope))**2
        for (t,p) in itertools.izip(true_dists, proj_dists) if p)
def slope_error(invslope):
    offsets = [1-(t/p*invslope) for (t,p)
               in itertools.izip(true_dists, proj_dists) if p]
    # Positive means t/p is smaller than slope.
    return sum(-o if o<0 else 5*o for o in offsets)
proj_over_true = scipy.optimize.fmin(slope_error, 1./average_slope)
print 'Best is %g' % (1./proj_over_true)

# Plot true vs. projected dist.

fig = plt.figure(1)

truedist = [numpy.linalg.norm(coords[w,:]-coords[central,:],2)
            for w, name in enumerate(ids)]
projdist = [numpy.linalg.norm(projected[w,:]-projected[central,:],2)
            for w, name in enumerate(ids)]
color = [1-(t/p*proj_over_true if p else 1) for t,p in itertools.izip(truedist, projdist)]

plt.scatter(truedist, projdist, c=color, cmap='jet', linewidths=0)
plt.plot([0,max(truedist)],[0,max(truedist)*proj_over_true], 'r--')
plt.savefig('errors.png')
fig.clear()

# Plot the starmap.

fig = plt.figure(2, figsize=(14,14))
plt.axes((0, 0, 1, 1))
plt.axis('equal')
plt.axis('off')

plt.scatter(projected[:,0], projected[:,1], c=color, cmap='jet', linewidths=0)

for w, name in enumerate(ids):
    x, y = projected[w,:]
    plt.text(x, y, str(name), size='xx-small')
plt.savefig('map.png')

