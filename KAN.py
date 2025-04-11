#!/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, f1_score, classification_report, roc_auc_score
from scipy.interpolate import CubicSpline, PchipInterpolator, Akima1DInterpolator,BSpline,interp1d
from sklearn.datasets import make_swiss_roll
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import OneHotEncoder
import copy

class MixedSplineLayer(nn.Module):
	"""A layer that applies a weighted mixture of multiple spline functions"""
	def __init__(self, num_knots=15, spline_types=('akima', 'b_spline', 'c_spline', 'cubic', 'pchip'), spline_order=5):
		super().__init__()
		self.num_knots = num_knots
		self.spline_types = spline_types
		self.knots = nn.Parameter(torch.linspace(-1, 1, num_knots), requires_grad=False)
		self.coeffs = nn.ParameterDict({
			spline: nn.Parameter(torch.randn(num_knots) * 0.1) for spline in spline_types
		})
		self.weights = nn.Parameter(torch.ones(len(spline_types)) / len(spline_types))
		self.spline_order  = spline_order

	def forward(self, x):
		device = x.device
		x = torch.clamp(x, min=self.knots[0].item(), max=self.knots[-1].item())
		x_np = x.cpu().detach().numpy()

		interpolations = []
		for spline, coeffs in self.coeffs.items():
			coeffs_np = coeffs.cpu().detach().numpy()
			knots_np = self.knots.cpu().numpy()
			if spline == 'akima':
				spline_func = Akima1DInterpolator(self.knots.cpu().numpy(), coeffs_np)
			elif spline == "b_spline":
				k = min(self.spline_order, self.num_knots - 1)
				extended_knots = np.concatenate(([knots_np[0]] * k, knots_np, [knots_np[-1]] * k))
				spline_func = BSpline(extended_knots, coeffs_np, k) 
			elif spline == "c_spline":
				t = np.linspace(0, 1, len(knots_np))
				spline_func = interp1d(t, coeffs_np, kind='cubic')
			elif spline == 'cubic':
				spline_func = CubicSpline(self.knots.cpu().numpy(), coeffs_np)
			elif spline == 'pchip':
				spline_func = PchipInterpolator(self.knots.cpu().numpy(), coeffs_np)
			else:
				raise ValueError(f"Unknown spline type: {spline}")

			interpolations.append(torch.tensor(spline_func(x_np), dtype=torch.float32).to(device))

		# Now we need to sum the splines together
		weights_softmax = torch.softmax(self.weights, dim=0)
		result = sum(w * interp for w, interp in zip(weights_softmax, interpolations))

		return result

class KANActivation(nn.Module):
	"""KAN activation function using a mixed spline approach"""
	def __init__(self, num_knots=15, spline_types=('cubic', 'pchip', 'akima'), spline_order=5):
		super().__init__()
		self.basis_weight = nn.Parameter(torch.randn(1))
		self.spline_weight = nn.Parameter(torch.ones(1))
		self.basis_function = nn.SiLU()
		self.spline_order = spline_order
		self.spline_layer = MixedSplineLayer(num_knots, spline_types=spline_types,spline_order=self.spline_order)
	def forward(self, x):
		# keep a residual connection with the splines 
		return self.basis_weight * self.basis_function(x) + self.spline_weight * self.spline_layer(x)

class KANLayer(nn.Module):
	"""Model a layer of learnable splines"""
	def __init__(self, input_dim, output_dim, num_knots=15, spline_types=('cubic', 'pchip', 'akima'),spline_order=5,dropout=.1):
		super().__init__()
		self.spline_order = spline_order
		self.activations = nn.ModuleList([KANActivation(num_knots, spline_types, self.spline_order) for _ in range(input_dim)])
		self.linear = nn.Linear(input_dim, output_dim)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		x = torch.stack([self.activations[i](x[:, i]) for i in range(x.shape[1])], dim=1)
		x = self.dropout(x)
		return self.linear(x)

class KAN(nn.Module):
	"""A KAN is like an MLP, but at each layer there is a learnable function on the edge
		Here we implement this by creating KANLayers and putting into a feedforward neural network
	"""
	def __init__(self, input_dim, output_dim, hidden_dim=50, num_layers=2, num_knots=10, spline_types=('cubic', 'pchip', 'akima'), spline_order=5,dropout=.1):
		super().__init__()
		self.spline_order = spline_order
		self.layers = nn.ModuleList([
			KANLayer(input_dim if i == 0 else hidden_dim, hidden_dim, num_knots, spline_types,self.spline_order) 
			for i in range(num_layers)
		])
		self.dropout = nn.Dropout(dropout)
		self.output_layer = nn.Linear(hidden_dim, output_dim)

	def forward(self, x):
		for layer in self.layers:
			x = layer(x)
		x = self.dropout(x)
		return self.output_layer(x)



def preprocess(X,y):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)
	X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=33)
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test)

	X_train,X_val, X_test = torch.tensor(X_train, dtype=torch.float32),torch.tensor(X_val,dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
	y_train, y_val, y_test = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1), torch.tensor(y_val,dtype=torch.float32).unsqueeze(1), torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
	return X_train,X_val,X_test,y_train,y_val,y_test

def shuffle_tensors(X, y,random_seed=44):
    """Shuffles two tensors in the same order."""
    random_seed =33*random_seed
    torch.manual_seed(random_seed)
    idx = torch.randperm(X.size()[0])  # Generate random permutation of indices
    X_shuffled = X[idx]  # Shuffle X using the indices
    y_shuffled = y[idx]  # Shuffle y using the same indices
    return X_shuffled, y_shuffled


def train(model, X_train,y_train, optimizer, loss_func, n_epochs=1000,validation_data=None):
	if validation_data is not None:
		X_val, y_val = validation_data

	best_val_loss = 1e6
	best_model=None
	best_epoch=0
	epochs_no_improve=0
	patience=100
	for epoch in range(n_epochs):
		model.train()
		optimizer.zero_grad()
		y_pred = model(X_train)
		loss = loss_func(y_pred, y_train)
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
		optimizer.step()
		if epoch % 50 == 0:
			print(f"Epoch {epoch}, Train Loss: {loss.item():.4f}")

		model.eval()
		with torch.no_grad():
			val_pred = model(X_val)
			val_loss = loss_func(val_pred, y_val).item()
		if (best_val_loss - val_loss) > 1e-10:
			best_val_loss = val_loss
			best_model = copy.deepcopy(model.state_dict())
			best_epoch = epoch
			epochs_no_improve=0
		else:
			epochs_no_improve+=1
		if epoch % 100 == 0:
			print(f"Epoch {epoch}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}") 
		if epochs_no_improve > patience:
			break
	print(f"Best validation loss: Epoch: {best_epoch}, Loss: {best_val_loss}")
	return best_model,best_val_loss

def make_model(X,y, hidden_dim=15,num_layers=10,lr=.001,loss_func=None, problem_type="classification",weight_decay=1e-4,dropout=.05,num_knots=15,spline_order=5,spline_types=('cubic', 'pchip', 'akima')):
        if problem_type not in ["classification","regression"]:
                raise ValueError("invalid problem type")
        elif problem_type == "classification":
                output_dim=len(y.unique())
                if loss_func is None:
                        if output_dim == 2:
                            loss_func = nn.BCEWithLogitsLoss()
                            output_dim=1
                        else:
                            loss_func = nn.CrossEntropyLoss()
        elif problem_type ==  "regression":
                output_dim=1
                if loss_func is None:
                        loss_func = nn.MSELoss()
        input_dim=X.shape[1]
        model = KAN(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim, num_layers=num_layers,dropout=dropout,spline_types=spline_types,spline_order=spline_order,num_knots=num_knots)
        if problem_type == "classification":
                optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        return model,optimizer,loss_func

def bin_dataset(y, bins=4):
	y = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='quantile').fit_transform(y.reshape(-1, 1)).astype(int).flatten()
	return y
	
def main():
	# test a variety of datasets with the network

	# load the friedman regression dataset
	X, y = make_friedman1(n_samples=10000, n_features=5, noise=1.0, random_state=42)
	X_train,X_val,X_test,y_train,y_val,y_test = preprocess(X,y)

	model, optimizer, loss_func = make_model(X_train,y_train,problem_type="regression",num_layers=4,dropout=.1,weight_decay=1e-3,spline_order=8)

	model_dict, _ = train(model, X_train, y_train, optimizer, loss_func, n_epochs=1200,validation_data=[X_val,y_val])
	model.load_state_dict(model_dict)

	with torch.no_grad():
		y_pred = model(X_test).detach().numpy()
	mse = mean_squared_error(y_test.numpy(), y_pred)
	print(f"Friedman MSE: {mse:.4f}")

	# load the swiss roll dataset and make into bins for classifying
	X, y = make_swiss_roll(n_samples=20000, noise=0.1, random_state=33)
	y = bin_dataset(y)
	X_train,X_val,X_test,y_train,y_val,y_test = preprocess(X,y)
	y_train = y_train.squeeze().long().flatten()
	y_val = y_val.squeeze().long().flatten()
	y_test = y_test.squeeze().long().flatten()

	model, optimizer, loss_func = make_model(X_train,y_train.squeeze().long(),loss_func=nn.CrossEntropyLoss())
	model_dict, _ = train(model, X_train, y_train, optimizer, loss_func, n_epochs=1200,validation_data=[X_val,y_val])
	model.load_state_dict(model_dict)


	model.eval()
	with torch.no_grad():
        	y_pred_probs = model(X_test)
        	y_pred = torch.argmax(y_pred_probs, dim=1).numpy()
	f1 = f1_score(y_test.numpy().flatten(),y_pred,average="macro")
	print(f"Swiss Roll F1: {f1:.4f}")
	print(classification_report(y_test.numpy().flatten(), y_pred))

	# load the breast cancer dataset
	breast_cancer = load_breast_cancer()
	X, y = breast_cancer.data, breast_cancer.target
	X_train,X_val,X_test,y_train,y_val,y_test = preprocess(X,y)

	y_train = y_train.squeeze().long().flatten()
	y_val = y_val.squeeze().long().flatten()
	y_test = y_test.squeeze().long().flatten()
    
	model, optimizer, loss_func = make_model(
		X_train, 
		y_train, 
		hidden_dim=20,  
		num_layers=6,   
		lr=0.001,
		problem_type="classification",
		dropout=.2,
		weight_decay=1e-6
	)
    
	model_dict, _ = train(model, X_train, y_train, optimizer, loss_func, n_epochs=1200,validation_data=[X_val,y_val])
	model.load_state_dict(model_dict)
	model.eval()
	with torch.no_grad():
		y_pred_probs = model(X_test)
		y_pred = torch.argmax(y_pred_probs, dim=1).numpy()
	f1 = f1_score(y_test.numpy().flatten(),y_pred,average="macro")
	print(f"Breast Cancer F1: {f1:.4f}")
	print(classification_report(y_test.numpy(), y_pred))
	auc = roc_auc_score(y_test.numpy(), y_pred_probs[:, 1].numpy())
	print(f"Breast Cancer AUC: {auc:.4f}")
    



if __name__ == "__main__":
	main()


