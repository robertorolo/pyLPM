import numpy as np
import sklearn.metrics
import plotly.graph_objs as go
import plotly.offline as pyo
import plotly.figure_factory as ff

def isotopic_arrays(workspace, grid, domains, variables, domain_nan = -99):
    """Generates an isotopic subset of the data
    
    Args:
        workspace (workspace object): workspace variable
        grid (str): grid name
        domains (str): labels name
        variables (list): variables names as string
    Returns:
        masked_x (array), masked_y (array), masked_z (array), masked_vars (nd array), masked_domains (array)
    """

    masks = []
    for var in variables:
        masks.append(np.isnan(workspace.grids[grid].properties[var][:]))

    masks.append(workspace.grids[grid].labels[domains][:]==domain_nan)

    mask = np.logical_or.reduce(np.array(masks))

    masked_domains = np.ma.array(workspace.grids[grid].labels[domains][:], mask = mask).compressed()
    masked_domains = masked_domains.astype(int)
    masked_x, masked_y, masked_z = np.ma.array(workspace.grids[grid].X[:], mask = mask).compressed(), np.ma.array(workspace.grids[grid].Y[:], mask = mask).compressed(), np.ma.array(workspace.grids[grid].Z[:], mask = mask).compressed()

    masked_vars = []
    for var in variables:
        masked_var = np.ma.array(workspace.grids[grid].properties[var][:], mask = mask)
        masked_vars.append(masked_var.compressed())

    return masked_x, masked_y, masked_z, np.array(masked_vars), masked_domains

def prediction_arrays(workspace, grid, dep_var, indep_vars, domains):
    """Generates a prediction array based on independent and dependent variables.
    
    Arguments:
        workspace {[workspace object]} -- workspace variable
        grid {[str]} -- grid name
        dep_var {[str]} -- dependent variable
        indep_vars {[str lst]} -- independent variables list
        domains {[str]} -- domain variable name
    
    Returns:
        masked_x (array), masked_y (array), masked_z (array), masked_vars (nd array), masked_domains (array)
    """

    ind_masks = []
    for var in indep_vars:
        ind_masks.append(np.isnan(workspace.grids[grid].properties[var][:]))

    ind_mask = np.logical_or.reduce(np.array(ind_masks))
    
    dep_mask = np.isnan(workspace.grids[grid].properties[dep_var][:])

    final_mask = []
    for index, value in enumerate(ind_mask):
        if value == 0 and dep_mask[index] == 1:
            final_mask.append(0)
        else:
            final_mask.append(1)

    masked_x, masked_y, masked_z = np.ma.array(workspace.grids[grid].X[:], mask = final_mask).compressed(), np.ma.array(workspace.grids[grid].Y[:], mask = final_mask).compressed(), np.ma.array(workspace.grids[grid].Z[:], mask = final_mask).compressed()

    masked_vars = []
    for var in indep_vars:
        masked_var = np.ma.array(workspace.grids[grid].properties[var][:], mask = final_mask)
        masked_vars.append(masked_var.compressed())

    masked_domain = np.ma.array(workspace.grids[grid].labels[domains][:], mask = final_mask).compressed()

    return masked_x, masked_y, masked_z, np.array(masked_vars), masked_domain

def model_evaluation_classification(kfolds, model, X, y):
    """evaluates classification models by kfolds validation
    
    Arguments:
        kfolds {[kfolds object]} -- kfolds object variable
        model {[ML model object]} -- model object variable
        X {[ndarray]} -- train variables
        y {[array]} -- target variable array
    """

    k = 1

    for train_index, test_index in kfolds.split(X, y):
        #print('fold {}'.format(k))
        table_lst = []
        table_lst.append(['fold {}'.format(k), 'accuracy', 'precision', 'recall', 'f1-score', 'support'])

        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        classifier = model.fit(X_train, y_train)
        prediction = classifier.predict(X_test)

        doms = np.unique(y_test).tolist()
        confusion_mtrx = sklearn.metrics.confusion_matrix(y_test, prediction, labels=doms).tolist()

        for dom in doms:
            
            sum_dict = {}
            hits = 0
            num_predicted = prediction.tolist().count(dom)
            should_predicted = y_test.tolist().count(dom)

            for index, val in enumerate(prediction):
                if val == dom and val == y_test[index]:
                    hits = hits + 1

            sum_dict['accuracy'] = 'no domain sample'if y_test.tolist().count(dom) == 0 else round(hits/(y_test.tolist().count(dom)),2)
            sum_dict['precision'] = 'zero predicted' if num_predicted ==0 else round(hits/num_predicted,2)
            sum_dict['recall'] = 'zero predicted' if should_predicted ==0 else round(hits/should_predicted,2)
            sum_dict['f1-score'] = 'zero predicted' if should_predicted ==0 or num_predicted ==0 else round((hits/num_predicted+hits/should_predicted)/2,2)
            sum_dict['support'] = y_test.tolist().count(dom)

            #print('domain {}'.format(dom), sum_dict)

            table_lst.append(['domain {}'.format(dom), sum_dict['accuracy'], sum_dict['precision'], sum_dict['recall'], sum_dict['f1-score'],sum_dict['support']])

        obj = ff.create_table(table_lst)
        pyo.iplot(obj)

        fig = ff.create_annotated_heatmap(confusion_mtrx, x=['dom {}'.format(dom) for dom in doms], y=['dom {}'.format(dom) for dom in doms],annotation_text=confusion_mtrx, colorscale='Jet', hoverinfo='z')       

        layout = {
        #'title':'Confusion matrix fold {} \n'.format(k),
        'xaxis':{'title':'Predicted domain','zeroline':False,'scaleanchor':'y'},
        'yaxis':{'title':'Actual domain','zeroline':False},
        'width':600,
        'height':600,
        }

        fig.layout.update(layout)

        pyo.iplot(fig) 

        k = k + 1

def model_evaluation_regression(kfolds, model, X, y):
    """Evaluates regression models by kfolds validation
    
    Arguments:
        kfolds {[kfolds object]} -- kfolds object variable
        model {[ML model object]} -- model object variable
        X {[ndarray]} -- train variables
        y {[array]} -- target variable array
    """

    k = 1
    table_lsts = [['Fold', 'Mean error', 'Max abs error', 'MSE', 'R squared']]

    for train_index, test_index in kfolds.split(X, y):

        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        classifier = model.fit(X_train, y_train)
        prediction = classifier.predict(X_test)

        sum_dict = {}

        error_mean = sklearn.metrics.mean_squared_error(prediction, y_test)
        error_abs_max = np.max(np.abs(prediction - y_test))
        correlation = np.corrcoef(prediction, y_test)[1][0]
        bias = np.mean(prediction - y_test)

        sum_dict['MSE'] = round(error_mean,2)
        sum_dict['correlation coefficient'] = round(correlation,2)
        sum_dict['max abs error'] = round(error_abs_max,2)
        sum_dict['mean error (bias)'] = round(bias, 2)

        #print('fold {}'.format(k),sum_dict)

        fold_lst = [k, sum_dict['mean error (bias)'], sum_dict['max abs error'], sum_dict['MSE'], sum_dict['correlation coefficient']]
        table_lsts.append(fold_lst)

        k = k + 1
    
    obj = ff.create_table(table_lsts)
    return pyo.iplot(obj)


def show_decision_tree(classifier_tree, var_names):
    n_nodes = classifier_tree.tree_.node_count
    children_left = classifier_tree.tree_.children_left
    children_right = classifier_tree.tree_.children_right
    feature = classifier_tree.tree_.feature
    threshold = classifier_tree.tree_.threshold
    # The tree structure can be traversed to compute various properties such
    # as the depth of each node and whether or not it is a leaf.
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    print("The binary tree structure has %s nodes and has "
          "the following tree structure:"
          % n_nodes)
    for i in range(n_nodes):
        if is_leaves[i]:
            print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
        else:
            print("%snode=%s test node: go to node %s if %s <= %s else to "
                  "node %s."
                  % (node_depth[i] * "\t",
                     i,
                     children_left[i],
                     var_names[feature[i]],
                     threshold[i],
                     children_right[i],
                     ))
    print()


def show_decision_path(classifier_tree, X, var_names):


    node_indicator = classifier_tree.decision_path(X=X.T)

    # Similarly, we can also have the leaves ids reached by each sample.

    leave_id = classifier_tree.apply(X_test)

    # Now, it's possible to get the tests that were used to predict a sample or
    # a group of samples. First, let's make it for the sample.

    sample_id = 0
    node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                        node_indicator.indptr[sample_id + 1]]

    print('Rules used to predict sample %s: ' % sample_id)
    for node_id in node_index:
        if leave_id[sample_id] == node_id:
            continue

        if (X[sample_id, feature[node_id]] <= threshold[node_id]):
            threshold_sign = "<="
        else:
            threshold_sign = ">"

        print("decision id node %s : samples: %s  features: %s (= %s) %s %s)"
              % (node_id,
                 sample_id,
                 var_names[feature[node_id]],
                 X[sample_id, feature[node_id]],
                 threshold_sign,
                 threshold[node_id]))


