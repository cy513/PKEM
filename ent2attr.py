
from config import args

if args.dataset != 'GDELT':
    id2entity, word2id = {}, {}
    with open('./data/{}/entity2id.txt'.format(args.dataset), 'r', encoding='utf-8') as f:
        for i, line in enumerate(f.readlines()):
            ent_str, id = line.strip().split("\t")
            id2entity[id] = ent_str

    count = 0
    eid2wid = []
    for id in range(len(id2entity.keys())):
        entity_str = id2entity[str(id)]
        if "(" in entity_str and ")" in entity_str:
            begin = entity_str.find('(')
            end = entity_str.find(')')
            w1 = entity_str[:begin].strip()
            w2 = entity_str[begin+1: end]
            if w1 not in word2id:
                word2id[w1] = count
                count += 1
            if w2 not in word2id:
                word2id[w2] = count
                count += 1
            eid2wid.append([str(id), "0", str(word2id[w1])])
            eid2wid.append([str(id), "1", str(word2id[w2])])
        else:
            if entity_str not in word2id:
                word2id[entity_str] = count
                count += 1
            eid2wid.append([str(id), "2", str(word2id[entity_str])])

    with open('./data/{}/entity2attributes.txt'.format(args.dataset), 'w', encoding='utf-8') as f:
        for line in eid2wid:
            f.write("\t".join(line)+'\n')

else:
    count = 0
    entmap = {}
    countrymap = {}
    with open('./data/GDELT/entity2id_country.txt', 'r', encoding='utf-8') as f:
        for i, line in enumerate(f.readlines()):
            ent, id, cty = line.strip().split('\t')
            entmap[ent] = cty
            if cty != 'NULL':
                if cty not in countrymap:
                    countrymap[cty] = count
                    count = count + 1
        ent_num = i

    count = 0
    with open('./data/GDELT/entity2attributes.txt', 'w', encoding='utf-8') as f:
        for entname in entmap:
            if entmap[entname] in countrymap:
                rel = '0'
                ctrnum = countrymap[entmap[entname]]
                ent2triples = [str(count), '1', str(ctrnum + ent_num)]
                f.write("\t".join(ent2triples) + '\n')
            else:
                rel = '0'
            ent2triples = [str(count), rel, str(count)]
            f.write("\t".join(ent2triples) + '\n')
            count = count + 1

