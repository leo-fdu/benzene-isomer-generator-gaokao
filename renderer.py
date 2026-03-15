from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from io import BytesIO
from PIL import Image

def render(smiles, width=600, height=600):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    
    # --- 新增的优化步骤 ---
    mol = Chem.RemoveHs(mol)
    Chem.rdDepictor.Compute2DCoords(mol) # 确保2D坐标计算准确
    # -------------------

    d2d = rdMolDraw2D.MolDraw2DCairo(width, height)
    opts = d2d.drawOptions()
    opts.useBWAtomPalette()
    opts.bondLineWidth = 2.5
    
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    # 直接返回 PIL Image，方便在 Streamlit 中 st.image(render(...)) 直接显示
    img_bytes = d2d.GetDrawingText()
    image = Image.open(BytesIO(img_bytes))
    image.load()
    return image
