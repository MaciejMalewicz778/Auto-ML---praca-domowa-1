def decodeParams(params):
    corrected_dict = { k.replace('classifier__', ''): v for k, v in params.items() }

    return corrected_dict

def decodeParamsArray(paramArray):
    newArr = []
    for elem in paramArray:
       corrected_dict = decodeParams(elem) 
       newArr.append(corrected_dict)

    return newArr

def getPipeline(model, num_cols, cat_cols):
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy='most_frequent')),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )

    pipe = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", model)]
    )

    return pipe