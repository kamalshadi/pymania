from py2neo import Node, Relationship, Graph
import numpy

graph = Graph(host="canopus.cc.gatech.edu",password='1234')


#################### reading from neo4j database #######################

# Connection categories

# categorize pairs for a subject
def getdata_st(sub,roi1,roi2):
    """
    Args:
        sub (int): Subject ID.
        roi1 (str): First ROI name.
        roi2 (str):Second ROI name.

    Returns:
        Object: The return value. True for success, False otherwise.
    """
    query = f'''match (n:ROI)-[r:NOS]->(m:ROI)
    where n.name="{roi1}" and m.name="{roi2}" and r.SUBJECT={sub}
    return n.name as n1,m.name as n2,r._length as _length,r._weight as _weight,r.border as border'''
    A = graph.run(query).data()
    return A[0]


def getdata_sts(sub,rois):
    roi_list = "['" + "','".join(rois) + "']"
    query = f'''MATCH (n:ROI)-[r:NOS]->(m:ROI)
                WHERE n.name IN {roi_list} AND m.name IN {roi_list} AND r.SUBJECT={sub}
                RETURN n.name as n1, m.name as n2, r._length as _length, r._weight as _weight, r.border as border'''
    out = graph.run(query).data()
    return out


def getdata_st_subs(st,subs):
    sub_list = "[" + ','.join(map(str, subs)) + "]"
    query = f'''MATCH (n:ROI)-[r:NOS]->(m:ROI)
                WHERE n.name='{st.roi1}' AND m.name='{st.roi2}' AND r.SUBJECT IN {sub_list}
                RETURN n.name as n1, m.name as n2, r._length as _length, r._weight as _weight, r.border as border'''
    out = graph.run(query).data()
    return out


def getdata_pair_subs(st,subs):
    sub_list = "[" + ','.join(map(str, subs)) + "]"
    query = f'''MATCH (n:ROI)-[r:NOS]->(m:ROI)
                WHERE n.name='{st.roi1}' AND m.name='{st.roi2}' AND r.SUBJECT IN {sub_list}
                RETURN n.name as n1, m.name as n2, r._length as _length, r._weight as _weight, r.border as border'''
    out = graph.run(query).data()
    query = f'''MATCH (n:ROI)-[r:NOS]->(m:ROI)
                WHERE n.name='{st.roi2}' AND m.name='{st.roi1}' AND r.SUBJECT IN {sub_list}
                RETURN n.name as n1, m.name as n2, r._length as _length, r._weight as _weight, r.border as border'''
    out.extend(graph.run(query).data())
    return out


#################### writing to neo4j database #######################
def write_connection(roi1, roi2, relation_type, attributes, run=True):
    """Write a connection to Neo4j database

    Args:
        roi1 (str): Source ROI
        roi2 (str): Destination ROI
        relation_type (str): Type of relationship to update
        attributes (dict): Dictionary of attributes to update
        run (bool): Whether to run the query or return the query

    Returns:
        str: Query generated if run is False
    """
    match = f"MATCH (n:ROI{{name:'{roi1}'}}), (m:ROI{{name:'{roi2}'}}) "
    attribs = f"CREATE (n)-[:{relation_type} {{"
    for key in attributes:
        if type(attributes[key]) == str:
            attribs += str(key) + ":'" + str(attributes[key]) + "', "
        elif type(attributes[key]) in [list, numpy.ndarray]:
            attribs += str(key) + ":[" + ','.join([str(x) for x in attributes[key]]) + "], "
        else:
            attribs += str(key) + ':' + str(attributes[key]) + ', '
    query = match + attribs[:-2] + '}]->(m)'
    if run:
        graph.run(query)
    else:
        return query


def update_connection(roi1, roi2, relation_type, subject, attributes, run=True):
    """Update a connection in Neo4j database

    Args:
        roi1 (str): Source ROI
        roi2 (str): Destination ROI
        relation_type (str): Type of relationship to update
        subject (int or str): Subject ID
        attributes (dict): Dictionary of attributes to update
        run (bool): Whether to run the query or return the query

    Returns:
        str: Query generated if run is False
    """
    if type(subject) != str:
        subject = str(subject)
    match = f"MATCH (n:ROI{{name:'{roi1}'}})-[r:{relation_type}]->(m:ROI{{name:'{roi2}'}}) WHERE r.SUBJECT={subject}"
    attribs = ' SET '
    for key in attributes:
        if type(attributes[key]) == str:
            attribs += 'r.' + str(key) + "='" + str(attributes[key]) + "', "
        elif type(attributes[key]) == list:
            attribs += 'r.' + str(key) + "=[" + ','.join([str(x) for x in attributes[key]]) + "], "
        else:
            attribs += 'r.' + str(key) + '=' + str(attributes[key]) + ', '
    query = match + attribs[:-2]
    if run:
        graph.run(query)
    else:
        return query
