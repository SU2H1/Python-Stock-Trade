xgboost_path = os.path.dirname(xgboost.__file__)
xgboost_lib = os.path.join(xgboost_path, 'lib', 'libxgboost.dylib')
datas += [(xgboost_lib, 'xgboost/lib')]
