from py2neo import Node, Relationship, Graph
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
    query = f'''match (n:ROI)-[r]->(m:ROI)
    where n.name="{roi1}" and m.name="{roi2}" and r.SUBJECT={sub}
    return n.name as n1,m.name as n2,r._length as _length,r._weight as _weight'''
    A = graph.run(query).data()
    return A[0]

def getdata_sts(sub,rois):
    out = []
    for roi1 in rois:
        for roi2 in rois:
            if roi1==roi2:continue
            out.append(getdata_st(sub,roi1,roi2))
    return out

def getdata_st_subs(st,subs):
    pass

def getdata_pair_subs(st,subs):
    pass

#################### writing to neo4j database #######################
