"""
Script de visualisation pour l'implémentation FASTSIM
Ce script génère des visualisations des résultats de l'algorithme FASTSIM
en utilisant matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import time

# Classes de l'implémentation FASTSIM fournie par l'utilisateur
class Inputs:
    def __init__(self, ux, uy, fx, fy, mx, my):
        self.ux = ux
        self.uy = uy
        self.fx = fx
        self.fy = fy
        self.mx = mx
        self.my = my
        self.Tx = 0.0
        self.Ty = 0.0

class Helper:
    @staticmethod
    def pressure(P, S, x0, x):
        return P - S * (x0 - x)

    @staticmethod
    def force(args_T, AR, P):
        return args_T + AR * P

class Subroutine:
    def __init__(self, args):
        self.args = args

    def sr_v1(self, dy, y):
        Px, Py = 0.0, 0.0
        x0 = np.sqrt(1.0 - y ** 2)
        dx = 2.0 * x0 / self.args.mx
        AR = dx * dy
        B = x0
        x = x0 - dx / 2.0
        Sx = self.args.ux - y * self.args.fx

        while -(x + B) < 0.0:
            Sy = self.args.uy + self.args.fy * (x0 + x) / 2.0
            Px = Helper.pressure(Px, Sx, x0, x)
            Py = Helper.pressure(Py, Sy, x0, x)
            P = np.sqrt(Px ** 2 + Py ** 2) / (1.0 - y ** 2 - x ** 2)

            if P > 1.0:
                Px /= P
                Py /= P
            
            self.args.Tx = Helper.force(self.args.Tx, AR, Px)
            self.args.Ty = Helper.force(self.args.Ty, AR, Py)
            x0 = x
            x -= dx

class Fastsim(Subroutine):
    def __init__(self, args):
        super().__init__(args)
        self.T = []
        # Ajout de structures pour stocker les résultats pour la visualisation
        self.x_points = []
        self.y_points = []
        self.px_values = []
        self.py_values = []
        self.pressure_values = []
        self.slip_status = []

    
    def v1(self, dy, TOL):
        # Réinitialiser les structures de données pour la visualisation
        self.x_points = []
        self.y_points = []
        self.px_values = []
        self.py_values = []
        self.pressure_values = []
        self.slip_status = []
        
        s = 1.0
        while s > -2.0:
            if abs(self.args.ux) > abs(self.args.fx) or abs(self.args.uy) > abs(self.args.fy):
                control = False
                dy = 2.0 / self.args.my
                s = -1.0
                ymi = -1.0
            else:
                control = True
                dy = (1.0 - (self.args.ux / self.args.fx * s)) / (
                    np.floor((1.0 - (self.args.ux / self.args.fx * s)) * self.args.my / 2.0) + 1.0)
                ymi = (self.args.ux / self.args.fx * s) + dy

            nk = 1
            y = (1.0 + dy / 2.0) * s
            while nk != 0:
                y -= dy * s
                if y * s < ymi and not control:
                    nk = 0
                elif y * s < ymi:
                    if TOL < (dy / 2.0):
                        dy /= 2.0
                        y += (dy / 2.0) * s
                        self.sr_v1(dy, y)
                        nk = 1
                    else:
                        self.sr_v1(dy, y)
                        nk = 0
                else:
                    self.sr_v1(dy, y)
            s -= 2.0

        self.T = [2.0 * -self.args.Tx / np.pi, 2.0 * self.args.Ty / np.pi]
        
        return {
            'Fx': self.T[0],
            'Fy': self.T[1],
            'x_points': np.array(self.x_points),
            'y_points': np.array(self.y_points),
            'px_values': np.array(self.px_values),
            'py_values': np.array(self.py_values),
            'pressure_values': np.array(self.pressure_values),
            'slip_status': np.array(self.slip_status)
        }

def plot_fastsim_results(results, title="FASTSIM Results", save_path=None):
    """
    Génère des visualisations des résultats de l'algorithme FASTSIM.
    
    Parameters:
    results -- Dictionnaire contenant les résultats de l'algorithme
    title -- Titre principal des graphiques
    save_path -- Chemin pour sauvegarder les figures (optionnel)
    """
    # Extraire les données
    x = results['x_points']
    y = results['y_points']
    px = results['px_values']
    py = results['py_values']
    pressure = results['pressure_values']
    slip = results['slip_status']
    
    # Créer une figure avec 4 sous-graphiques
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(title, fontsize=16)
    
    # Définir les limites pour tous les graphiques
    x_min, x_max = -1.1, 1.1
    y_min, y_max = -1.1, 1.1
    
    # 1. Distribution de pression normale
    sc1 = axs[0, 0].scatter(x, y, c=pressure, cmap='viridis', s=10, alpha=0.7)
    axs[0, 0].set_title('Normal Pressure Distribution')
    axs[0, 0].set_xlabel('x')
    axs[0, 0].set_ylabel('y')
    axs[0, 0].set_xlim(x_min, x_max)
    axs[0, 0].set_ylim(y_min, y_max)
    axs[0, 0].grid(True)
    # Ajouter un cercle pour représenter la zone de contact
    circle = plt.Circle((0, 0), 1, fill=False, color='red', linestyle='--')
    axs[0, 0].add_patch(circle)
    fig.colorbar(sc1, ax=axs[0, 0], label='Pressure')
    
    # 2. Statut de glissement (slip/stick)
    sc2 = axs[0, 1].scatter(x, y, c=slip, cmap='coolwarm', s=10, alpha=0.7)
    axs[0, 1].set_title('Slip Status (Red = Slip, Blue = Stick)')
    axs[0, 1].set_xlabel('x')
    axs[0, 1].set_ylabel('y')
    axs[0, 1].set_xlim(x_min, x_max)
    axs[0, 1].set_ylim(y_min, y_max)
    axs[0, 1].grid(True)
    # Ajouter un cercle pour représenter la zone de contact
    circle = plt.Circle((0, 0), 1, fill=False, color='black', linestyle='--')
    axs[0, 1].add_patch(circle)
    
    # 3. Contrainte tangentielle en x
    sc3 = axs[1, 0].scatter(x, y, c=px, cmap='coolwarm', s=10, alpha=0.7)
    axs[1, 0].set_title('Tangential Stress (x-direction)')
    axs[1, 0].set_xlabel('x')
    axs[1, 0].set_ylabel('y')
    axs[1, 0].set_xlim(x_min, x_max)
    axs[1, 0].set_ylim(y_min, y_max)
    axs[1, 0].grid(True)
    # Ajouter un cercle pour représenter la zone de contact
    circle = plt.Circle((0, 0), 1, fill=False, color='black', linestyle='--')
    axs[1, 0].add_patch(circle)
    fig.colorbar(sc3, ax=axs[1, 0], label='Stress (x)')
    
    # 4. Contrainte tangentielle en y
    sc4 = axs[1, 1].scatter(x, y, c=py, cmap='coolwarm', s=10, alpha=0.7)
    axs[1, 1].set_title('Tangential Stress (y-direction)')
    axs[1, 1].set_xlabel('x')
    axs[1, 1].set_ylabel('y')
    axs[1, 1].set_xlim(x_min, x_max)
    axs[1, 1].set_ylim(y_min, y_max)
    axs[1, 1].grid(True)
    # Ajouter un cercle pour représenter la zone de contact
    circle = plt.Circle((0, 0), 1, fill=False, color='black', linestyle='--')
    axs[1, 1].add_patch(circle)
    fig.colorbar(sc4, ax=axs[1, 1], label='Stress (y)')
    
    # Ajouter les informations sur les forces
    plt.figtext(0.5, 0.01, f"Fx = {results['Fx']:.4f}, Fy = {results['Fy']:.4f}", 
                ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
    
    return fig

def plot_vector_field(results, title="Tangential Stress Vector Field", save_path=None):
    """
    Génère une visualisation du champ de vecteurs des contraintes tangentielles.
    
    Parameters:
    results -- Dictionnaire contenant les résultats de l'algorithme
    title -- Titre du graphique
    save_path -- Chemin pour sauvegarder la figure (optionnel)
    """
    # Extraire les données
    x = results['x_points']
    y = results['y_points']
    px = results['px_values']
    py = results['py_values']
    slip = results['slip_status']
    
    # Créer une figure
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.suptitle(title, fontsize=16)
    
    # Définir les limites
    x_min, x_max = -1.1, 1.1
    y_min, y_max = -1.1, 1.1
    
    # Calculer la magnitude des vecteurs pour la coloration
    magnitude = np.sqrt(px**2 + py**2)
    
    # Tracer les vecteurs avec coloration selon le statut de glissement
    colors = np.array(['blue' if not s else 'red' for s in slip])
    
    # Utiliser quiver pour tracer les vecteurs
    q = ax.quiver(x, y, px, py, color=colors, scale=20, width=0.003)
    
    # Ajouter une légende
    ax.quiverkey(q, X=0.85, Y=1.05, U=1, label='Unit Vector', labelpos='E')
    
    # Ajouter un cercle pour représenter la zone de contact
    circle = plt.Circle((0, 0), 1, fill=False, color='black', linestyle='--')
    ax.add_patch(circle)
    
    # Configurer les axes
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True)
    
    # Ajouter une légende pour les couleurs
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Stick'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Slip')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Ajouter les informations sur les forces
    plt.figtext(0.5, 0.01, f"Fx = {results['Fx']:.4f}, Fy = {results['Fy']:.4f}", 
                ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path)
        print(f"Vector field figure saved to {save_path}")
    
    return fig

def plot_3d_surface(results, title="3D Surface Plot of Tangential Stress", save_path=None):
    """
    Génère une visualisation 3D de la distribution des contraintes tangentielles.
    
    Parameters:
    results -- Dictionnaire contenant les résultats de l'algorithme
    title -- Titre du graphique
    save_path -- Chemin pour sauvegarder la figure (optionnel)
    """
    # Extraire les données
    x = results['x_points']
    y = results['y_points']
    px = results['px_values']
    py = results['py_values']
    
    # Calculer la magnitude des contraintes tangentielles
    magnitude = np.sqrt(px**2 + py**2)
    
    # Créer une figure 3D
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Tracer la surface 3D
    scatter = ax.scatter(x, y, magnitude, c=magnitude, cmap='viridis', s=10)
    
    # Configurer les axes
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Tangential Stress Magnitude')
    ax.set_title(title)
    
    # Ajouter une barre de couleur
    fig.colorbar(scatter, ax=ax, label='Stress Magnitude')
    
    # Ajouter les informations sur les forces
    plt.figtext(0.5, 0.01, f"Fx = {results['Fx']:.4f}, Fy = {results['Fy']:.4f}", 
                ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path)
        print(f"3D surface figure saved to {save_path}")
    
    return fig

def run_and_visualize_test_case(test_id, ux, uy, fx, fy, mx=5, my=0.4, tol=3):
    """
    Exécute un cas de test et génère des visualisations.
    
    Parameters:
    test_id -- Identifiant du test
    ux, uy -- Creepages longitudinal et latéral
    fx, fy -- Paramètres liés au spin
    mx, my -- Nombre de pas dans les directions x et y
    tol -- Limite inférieure de la largeur de tranche
    """
    print(f"\n=== Exécution et visualisation du cas de test {test_id} ===")
    print(f"Paramètres: ux={ux}, uy={uy}, fx={fx}, fy={fy}, mx={mx}, my={my}, tol={tol}")
    
    # Initialiser l'algorithme FASTSIM
    args = Inputs(ux, uy, fx, fy, mx, my)
    fastsim = Fastsim(args)
    
    # Exécuter l'algorithme et mesurer le temps
    start_time = time.time()
    results = fastsim.v1(2, tol)  # 0.4 est la valeur de b (demi-axe latéral)
    end_time = time.time()
    
    elapsed_time = (end_time - start_time) * 1e6  # Conversion en microsecondes
    print(f"Temps d'exécution: {elapsed_time:.2f} microsecondes")
    print(f"Résultats: Fx={results['Fx']:.4f}, Fy={results['Fy']:.4f}")
    
    # Générer les visualisations
    title = f"FASTSIM Results - Test {test_id}: ux={ux}, uy={uy}, fx={fx}, fy={fy}"
    
    # 1. Graphiques principaux
    fig1 = plot_fastsim_results(results, title=title, 
                               save_path=f"results/fastsim_results_{test_id}.png")
    
    # 2. Champ de vecteurs
    fig2 = plot_vector_field(results, title=f"Tangential Stress Vector Field - Test {test_id}",
                            save_path=f"results/fastsim_vectors_{test_id}.png")
    
    # 3. Surface 3D
    fig3 = plot_3d_surface(results, title=f"3D Surface Plot - Test {test_id}",
                          save_path=f"results/fastsim_3d_{test_id}.png")
    
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    
    return results

def run_all_visualizations():
    """
    Exécute et visualise tous les cas de test du papier de Kalker.
    """
    # Augmenter la résolution pour de meilleures visualisations
    mx = 5
    my = 0.4
    
    # Cas de test A2.1: ux=0, uy=-2, fx=2, fy=4
    run_and_visualize_test_case('A2.1', ux=0, uy=-2, fx=2, fy=4, mx=mx, my=my, tol=3)
    
    # Cas de test A2.2: ux=0, uy=-2, fx=2, fy=4
    #run_and_visualize_test_case('A2.2', ux=1.0, uy=-2, fx=2, fy=1, mx=mx, my=my, tol=0.09)
    
    # Cas de test A2.3: ux=1, uy=-2, fx=2, fy=4
    #run_and_visualize_test_case('A2.3', ux=1, uy=-2, fx=2, fy=4, mx=mx, my=my, tol=0.09)
    
    # Cas de test A2.4: ux=1, uy=-2, fx=2, fy=4
    #run_and_visualize_test_case('A2.4', ux=1, uy=-2, fx=2, fy=4, mx=mx, my=my, tol=3)
    
    # Cas de test A2.5: ux=1, uy=-2, fx=2, fy=1
    #run_and_visualize_test_case('A2.5', ux=1, uy=-2, fx=2, fy=1, mx=mx, my=my, tol=0.09)
    
    print("\nToutes les visualisations ont été générées avec succès!")

if __name__ == "__main__":
    run_all_visualizations()
