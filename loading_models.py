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
    baseline.final_model = pickle.load(open('final_models/baseline.pkl', 'rb'))

    bsknn = Model()
    bsknn.name = 'bsknn'
    bsknn.model = pickle.load(open('models/bsknn.pkl', 'rb'))
    bsknn.final_model = pickle.load(open('final_models/bsknn.pkl', 'rb'))

    so = Model()
    so.name = 'so'
    so.model = pickle.load(open('models/so.pkl', 'rb'))
    so.final_model = pickle.load(open('final_models/so.pkl', 'rb'))

    svd = Model()
    svd.name = 'svd'
    svd.model = pickle.load(open('models/svd.pkl', 'rb'))
    svd.final_model = pickle.load(open('final_models/svd.pkl', 'rb'))

    als = Model()
    als.name = 'als'
    als.model = pickle.load(open('models/als.pkl', 'rb'))
    als.final_model = pickle.load(open('final_models/als.pkl', 'rb'))

    sgd = Model()
    sgd.name = 'sgd'
    sgd.model = pickle.load(open('models/sgd.pkl', 'rb'))
    sgd.final_model = pickle.load(open('final_models/sgd.pkl', 'rb'))

    globalmean = Model()
    globalmean.name = 'globalmean'
    globalmean.model = pickle.load(open('models/globalmean.pkl', 'rb'))
    globalmean.final_model = pickle.load(open('final_models/globalmean.pkl', 'rb'))

    usermean = Model()
    usermean.name = 'usermean'
    usermean.model = pickle.load(open('models/usermean.pkl', 'rb'))
    usermean.final_model = pickle.load(open('final_models/usermean.pkl', 'rb'))

    itemmean = Model()
    itemmean.name = 'itemmean'
    itemmean.model = pickle.load(open('models/itemmean.pkl', 'rb'))
    itemmean.final_model = pickle.load(open('final_models/itemmean.pkl', 'rb'))

    basicknn = Model()
    basicknn.name = 'basicknn'
    basicknn.model = pickle.load(open('models/basicknn.pkl', 'rb'))
    basicknn.final_model = pickle.load(open('final_models/basicknn.pkl', 'rb'))

    svdpp = Model()
    svdpp.name = 'svdpp'
    svdpp.model = pickle.load(open('models/svdpp.pkl', 'rb'))
    svdpp.final_model = pickle.load(open('final_models/svdpp.pkl', 'rb'))

    globalmedian = Model()
    globalmedian.name = 'globalmedian'
    globalmedian.model = pickle.load(open('models/globalmedian.pkl', 'rb'))
    globalmedian.final_model = pickle.load(open('final_models/globalmedian.pkl', 'rb'))

    usermedian = Model()
    usermedian.name = 'usermedian'
    usermedian.model = pickle.load(open('models/usermedian.pkl', 'rb'))
    usermedian.final_model = pickle.load(open('final_models/usermedian.pkl', 'rb'))

    itemmedian = Model()
    itemmedian.name = 'itemmedian'
    itemmedian.model = pickle.load(open('models/itemmedian.pkl', 'rb'))
    itemmedian.final_model = pickle.load(open('final_models/itemmedian.pkl', 'rb'))

    models = [globalmean, usermean, itemmean, baseline, bsknn, basicknn, so, svd, als, sgd, svdpp, globalmedian,
              usermedian, itemmedian]

    return models