import pandas as pd
from skopt import gp_minimize
from skopt.space import Real
from skopt.callbacks import VerboseCallback
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementProperty
from catboost import CatBoostRegressor
import joblib

model = CatBoostRegressor()
model.load_model('catboost_energy_model.cbm')
feature_names = joblib.load('feature.pkl') 

results_df = pd.DataFrame(columns=['Composition', 'Energy Difference'])

def objective(params):
    Li, Ni, Fe, Cu, Co, Zr, Mn = [round(p, 2) for p in params]

    if abs(Li + Ni + Fe + Cu + Co + Zr + Mn - 1.0) > 1e-6:
        return 1000

    composition = f"NaLi{Li:.2f}Zr{Zr:.2f}Fe{Fe:.2f}Co{Co:.2f}Ni{Ni:.2f}Cu{Cu:.2f}Mn{Mn:.2f}O2"

    comp_df = pd.DataFrame({'formula': [composition]})
    comp_df = StrToComposition().featurize_dataframe(comp_df, "formula", ignore_errors=True)
    comp_df = ep_feat.featurize_dataframe(comp_df, col_id="composition", ignore_errors=True)

    mole_fractions = [Li, Zr, Fe, Co, Ni, Cu, Mn] 
    total = sum(mole_fractions)
    normalized_mole_fractions = [mf / total for mf in mole_fractions]

    for i, mf in enumerate(normalized_mole_fractions):
        comp_df[f'MoleFraction_{i + 1}'] = round(mf, 2) 

    comp_df = comp_df.drop(columns=['formula', 'composition'])

    X_new = comp_df[feature_names]

    y_pred = model.predict(X_new)

    results_df.loc[len(results_df)] = [composition, y_pred[0]]

    results_df.to_csv('optimization_results.csv', index=False)

    print(f"Iteration composition: {composition}, Function value obtained: {y_pred[0]}")

    return y_pred[0]


# space
space = [
    Real(0.04, 0.07, name='Li'),
    Real(0.2, 0.3, name='Ni'),
    Real(0.01, 0.2, name='Fe'),
    Real(0.02, 0.05, name='Cu'),
    Real(0.02, 0.05, name='Co'),
    Real(0.01, 0.06, name='Zr'),
    Real(0.3, 0.45, name='Mn'),
]

if __name__ == "__main__":
    ep_feat = ElementProperty.from_preset(preset_name="magpie")

    verbose_callback = VerboseCallback(n_total=100)

    result = gp_minimize(objective, space, n_calls=100, random_state=0, callback=[verbose_callback])

    best_params = result.x
    best_value = result.fun

    print(
        f"best: Li={best_params[0]:.2f}, Ni={best_params[1]:.2f}, Fe={best_params[2]:.2f}, Cu={best_params[3]:.2f}, Co={best_params[4]:.2f}, Zr={best_params[5]:.2f}, Mn={best_params[6]:.2f}")
    print(f"best_energy_difference: {best_value}")

    top_10_results = results_df.nsmallest(10, 'Energy Difference')

    print("\nTop 10 compositions and their corresponding energy differences:")
    for _, row in top_10_results.iterrows():
        print(f"Composition: {row['Composition']}, Energy difference: {row['Energy Difference']}")
