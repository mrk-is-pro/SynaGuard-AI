// Train the autoencoder on the sample once, so the dashboard loads the sample instantly (run: node make_model.js)
const SG = require('./engine.js'), fs = require('fs');
const { head, rows } = SG.parseCSV(fs.readFileSync(__dirname + '/generated/sample.csv', 'utf8'));
const C = SG.clean(head, rows, SG.guess(head)), F = SG.features(C), t = Date.now();
const fit = SG.fit(F, { ep: 40, ep1: 40 });
fs.writeFileSync(__dirname + '/generated/model.json', JSON.stringify(SG.pack(fit)));
console.log('model.json written; features', F.names.length, 'threshold', fit.thr.toFixed(3), `(${Date.now() - t} ms)`);
