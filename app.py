import os
import requests
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, Draw, AllChem

st.set_page_config(
    page_title="AI-GROMACS",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 AI-GROMACS")
st.subheader("AI-Powered Molecular Analysis Platform")

st.write(
    "A platform combining Chemistry, GROMACS Molecular Dynamics, "
    "Data Analysis, and Artificial Intelligence."
)

st.divider()

# ----------------------------------------------------
# Section 1: Chemistry Analysis
# ----------------------------------------------------
st.header("🧪 1. Chemistry Analysis & 2D Inspection")

smiles = st.text_input(
    "Enter a SMILES structure",
    value="CC(=O)Oc1ccccc1C(=O)O",
    placeholder="Example: CC(=O)Oc1ccccc1C(=O)O"
)

if smiles:
    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        st.error("Invalid SMILES structure. Please check your input.")
    else:
        st.success("Molecule successfully recognized!")

        molecular_weight = Descriptors.MolWt(molecule)
        logp = Crippen.MolLogP(molecule)
        hbd = Lipinski.NumHDonors(molecule)
        hba = Lipinski.NumHAcceptors(molecule)
        tpsa = Descriptors.TPSA(molecule)
        rotatable_bonds = Lipinski.NumRotatableBonds(molecule)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Molecular Weight", f"{molecular_weight:.2f} g/mol")
            st.metric("H-Bond Donors", str(hbd))

        with col2:
            st.metric("LogP", f"{logp:.2f}")
            st.metric("H-Bond Acceptors", str(hba))

        with col3:
            st.metric("TPSA", f"{tpsa:.2f} Å²")
            st.metric("Rotatable Bonds", str(rotatable_bonds))

        st.divider()

        st.subheader("2D Molecular Structure")
        st.image(
            Draw.MolToImage(molecule, size=(400, 300)),
            caption="2D Chemical Representation"
        )

        st.divider()

        # ----------------------------------------------------
        # Section 2: 3D Conformation & Ligand MD Preparation
        # ----------------------------------------------------
        st.header("⚙️ 2. Ligand 3D Conformation & Preparation")
        st.write("Convert 2D structure to a 3D coordinate system with explicit hydrogens.")

        ligand_name = st.text_input("Ligand Identifier / File Name", value="ligand")

        if st.button("Generate 3D Coordinates & Save Ligand PDB"):
            with st.spinner("Adding Hydrogens, generating 3D coordinates, and minimizing energy..."):
                mol_3d = Chem.AddHs(molecule)
                embed_status = AllChem.EmbedMolecule(mol_3d, AllChem.ETKDG())

                if embed_status != 0:
                    st.warning("Standard embedding encountered issues; running random coordinate fallback...")
                    AllChem.EmbedMolecule(mol_3d, useRandomCoords=True)

                AllChem.MMFFOptimizeMolecule(mol_3d, maxIters=500)

                output_dir = os.path.join(os.getcwd(), "data", "raw")
                os.makedirs(output_dir, exist_ok=True)
                pdb_filepath = os.path.join(output_dir, f"{ligand_name}.pdb")

                Chem.MolToPDBFile(mol_3d, pdb_filepath)
                pdb_block = Chem.MolToPDBBlock(mol_3d)

                st.success(f"Ligand saved: `{pdb_filepath}`")

                st.download_button(
                    label="⬇️ Download Ligand PDB",
                    data=pdb_block,
                    file_name=f"{ligand_name}.pdb",
                    mime="chemical/x-pdb"
                )

                with st.expander("Preview Ligand PDB Coordinates"):
                    st.code(pdb_block, language="text")

st.divider()

# ----------------------------------------------------
# Section 3: Target Protein Retrieval & Preparation
# ----------------------------------------------------
st.header("🦠 3. Target Protein Retrieval & Preparation")
st.write("Fetch standard RCSB/PDBe structures and filter out crystallographic water molecules (`HOH`).")

col_prot1, col_prot2 = st.columns([1, 2])

with col_prot1:
    pdb_id = st.text_input("Enter 4-letter PDB ID", value="1HSG", max_chars=4).strip().lower()
    fetch_btn = st.button("Fetch & Clean Protein")

if fetch_btn:
    if len(pdb_id) != 4:
        st.error("Please enter a valid 4-character PDB code.")
    else:
        with st.spinner(f"Fetching {pdb_id.upper()} from Protein Data Bank..."):
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            }
            
            # Primary mirror: RCSB, Secondary mirror: PDBe
            urls = [
                f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb",
                f"https://www.ebi.ac.uk/pdbe/entry-files/download/pdb{pdb_id}.ent"
            ]
            
            raw_text = None
            for url in urls:
                try:
                    res = requests.get(url, headers=headers, timeout=15)
                    if res.status_code == 200 and "ATOM" in res.text:
                        raw_text = res.text
                        break
                except requests.RequestException:
                    continue

            if raw_text:
                raw_pdb_lines = raw_text.splitlines()

                # Filter out crystallographic waters ('HOH')
                cleaned_lines = [
                    line for line in raw_pdb_lines 
                    if not (line.startswith("HETATM") and "HOH" in line)
                ]

                output_dir = os.path.join(os.getcwd(), "data", "raw")
                os.makedirs(output_dir, exist_ok=True)
                cleaned_filepath = os.path.join(output_dir, f"{pdb_id.upper()}_clean.pdb")

                with open(cleaned_filepath, "w") as f:
                    f.write("\n".join(cleaned_lines))

                st.success(f"Protein `{pdb_id.upper()}` downloaded and cleaned: `{cleaned_filepath}`")

                atom_count = sum(1 for l in cleaned_lines if l.startswith("ATOM"))
                hetatm_count = sum(1 for l in cleaned_lines if l.startswith("HETATM"))

                m1, m2 = st.columns(2)
                m1.metric("Protein Atoms (ATOM)", atom_count)
                m2.metric("Non-Water Heteroatoms (HETATM)", hetatm_count)

                with st.expander("Preview Cleaned Protein PDB (First 50 lines)"):
                    st.code("\n".join(cleaned_lines[:50]), language="text")
            else:
                st.error(f"Failed to fetch PDB `{pdb_id.upper()}` from RCSB and PDBe mirrors. Please check connection or ID.")

st.divider()
st.info("Version 1 — Chemistry, 3D Conformation & Protein Preparation")