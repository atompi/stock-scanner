from flask import Flask, jsonify, render_template, request

from stock_analyzer import StockAnalyzer

app = Flask(__name__)
analyzer = StockAnalyzer()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        stock_code = data.get('stock_code')
        if not stock_code:
            return jsonify({'error': '请提供股票代码'}), 400

        result = analyzer.analyze_stock(stock_code)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/batch-analyze', methods=['POST'])
def batch_analyze():
    try:
        data = request.json
        stock_list = data.get('stock_list', [])
        if not stock_list:
            return jsonify({'error': '请提供股票代码列表'}), 400

        results = analyzer.scan_market(stock_list)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
