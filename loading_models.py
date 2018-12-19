import pickle

class Model:
    name = ''
    rmse = 0.0
    model = ''
    final_model = ''

def loadModels():
    baseline = Model()
    baseline.name = 'baseline'
    baseline.model = pickle.load(open('models/baseline.pkl', 'rb'))
    baseline.final_model = pickle.load(open('final_models/baseline_final.pkl', 'rb'))

    bsknn = Model()
    bsknn.name = 'bsknn'
    bsknn.model = pickle.load(open('models/bsknn.pkl', 'rb'))
    bsknn.final_model = pickle.load(open('final_models/bsknn_final.pkl', 'rb'))

    so = Model()
    so.name = 'so'
    so.model = pickle.load(open('models/so.pkl', 'rb'))
    so.final_model = pickle.load(open('final_models/so_final.pkl', 'rb'))

    svd = Model()
    svd.name = 'svd'
    svd.model = pickle.load(open('models/svd.pkl', 'rb'))
    svd.final_model = pickle.load(open('final_models/svd_final.pkl', 'rb'))

    als = Model()
    als.name = 'als'
    als.model = pickle.load(open('models/als.pkl', 'rb'))
    als.final_model = pickle.load(open('final_models/als_final.pkl', 'rb'))

    sgd = Model()
    sgd.name = 'sgd'
    sgd.model = pickle.load(open('models/sgd.pkl', 'rb'))
    sgd.final_model = pickle.load(open('final_models/sgd_final.pkl', 'rb'))

    globalmean = Model()
    globalmean.name = 'globalmean'
    globalmean.model = pickle.load(open('models/globalmean.pkl', 'rb'))
    globalmean.final_model = pickle.load(open('final_models/globalmean_final.pkl', 'rb'))

    usermean = Model()
    usermean.name = 'usermean'
    usermean.model = pickle.load(open('models/usermean.pkl', 'rb'))
    usermean.final_model = pickle.load(open('final_models/usermean_final.pkl', 'rb'))

    itemmean = Model()
    itemmean.name = 'itemmean'
    itemmean.model = pickle.load(open('models/itemmean.pkl', 'rb'))
    itemmean.final_model = pickle.load(open('final_models/itemmean_final.pkl', 'rb'))

    basicknn = Model()
    basicknn.name = 'basicknn'
    basicknn.model = pickle.load(open('models/basicknn.pkl', 'rb'))
    basicknn.final_model = pickle.load(open('final_models/basicknn_final.pkl', 'rb'))

    svdpp = Model()
    svdpp.name = 'svdpp'
    svdpp.model = pickle.load(open('models/svdpp.pkl', 'rb'))
    svdpp.final_model = pickle.load(open('final_models/svdpp_final.pkl', 'rb'))

    globalmedian = Model()
    globalmedian.name = 'globalmedian'
    globalmedian.model = pickle.load(open('models/globalmedian.pkl', 'rb'))
    globalmedian.final_model = pickle.load(open('final_models/globalmedian_final.pkl', 'rb'))

    usermedian = Model()
    usermedian.name = 'usermedian'
    usermedian.model = pickle.load(open('models/usermedian.pkl', 'rb'))
    usermedian.final_model = pickle.load(open('final_models/usermedian_final.pkl', 'rb'))

    itemmedian = Model()
    itemmedian.name = 'itemmedian'
    itemmedian.model = pickle.load(open('models/itemmedian.pkl', 'rb'))
    itemmedian.final_model = pickle.load(open('final_models/itemmedian_final.pkl', 'rb'))

    models = [globalmean, usermean, itemmean, baseline, bsknn, basicknn, so, svd, als, sgd, svdpp, globalmedian,
              usermedian, itemmedian]

    return models