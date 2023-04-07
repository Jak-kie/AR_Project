import os

class OBJ:

    @classmethod
    def loadMaterial(cls, filename):
        contents = {}
        mtl = None
        dirname = os.path.dirname(filename)

        for line in open(filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'newmtl':
                mtl = contents[values[1]] = {}
            elif mtl is None:
                raise ValueError("mtl file doesn't start with newmtl stmt")
            elif values[0] == 'map_Kd':
                # load the texture referred to by this declaration
                mtl[values[0]] = values[1]
                imagefile = os.path.join(dirname, mtl['map_Kd'])
                mtl['texture_Kd'] = cls.loadTexture(imagefile)
            else:
                mtl[values[0]] = list(map(float, values[1:]))
        return contents


    def __init__(self, filename, swapyz=False):
        """Loads a Wavefront OBJ file. """
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []
        material = None
        dirname = os.path.dirname(filename)

        for line in open(filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.vertices.append(v)
            elif values[0] == 'vn':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.normals.append(v)
            elif values[0] == 'vt':
                self.texcoords.append(map(float, values[1:3]))
            elif values[0] == 'mtllib':
                self.mtl = self.loadMaterial(os.path.join(dirname, values[1]))
            elif values[0] in ('usemtl', 'usemat'):
                material = values[1]
            elif values[0] == 'f':
                """
                e.g. f 2/1/1 4/2/2 1/3/1
                     f v1[/vt1][/vn1]
                     v = vertex coord
                     vt = vertex texture
                     vn = vertex normal
                """
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:                    # v = 2/1/1         values = 2/1/1 4/2/2 1/3/1
                    w = v.split('/')                    # w = [2,1,1]
                    face.append(int(w[0]))              # aggiungo 2, superfluo convertirlo ad INT ma sia mai che ci siano errori
                    if len(w) >= 2 and len(w[1]) > 0:   # esiste un valore vt, e viene aggiunto
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(0)             # non esiste un valore vt, e quindi aggiungiamo 0
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))         # esiste un valore vn, e viene aggiunto
                    else:
                        norms.append(0)                 # non esiste un valore vn, e quindi aggiungiamo 0
                self.faces.append((face, norms, texcoords, material))