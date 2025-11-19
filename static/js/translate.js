(function(){
    // Simple cookie helpers
    function setCookie(name, value, days) {
        var expires = "";
        if (days) {
            var date = new Date();
            date.setTime(date.getTime() + (days*24*60*60*1000));
            expires = "; expires=" + date.toUTCString();
        }
        document.cookie = name + "=" + (value || "")  + expires + "; path=/";
    }
    function getCookie(name) {
        var nameEQ = name + "=";
        var ca = document.cookie.split(';');
        for(var i=0;i < ca.length;i++) {
            var c = ca[i];
            while (c.charAt(0)==' ') c = c.substring(1,c.length);
            if (c.indexOf(nameEQ) == 0) return c.substring(nameEQ.length,c.length);
        }
        return null;
    }

    // Create floating language selector UI
    function createControl(){
        if(document.getElementById('translate-control')) return;
        var container = document.createElement('div');
        container.id = 'translate-control';
        container.style.position = 'fixed';
        container.style.bottom = '18px';
        container.style.right = '18px';
        container.style.zIndex = '99999';
        container.style.background = 'rgba(255,255,255,0.95)';
        container.style.border = '1px solid #e0e0e0';
        container.style.padding = '6px 8px';
        container.style.borderRadius = '28px';
        container.style.boxShadow = '0 2px 8px rgba(0,0,0,0.12)';
        container.style.fontFamily = 'Arial, sans-serif';
        container.style.display = 'flex';
        container.style.alignItems = 'center';
        container.style.gap = '8px';

        var label = document.createElement('span');
        label.style.fontSize = '13px';
        label.style.color = '#333';
        label.textContent = 'Language';

        var select = document.createElement('select');
        select.id = 'site-lang-select';
        select.style.border = 'none';
        select.style.background = 'transparent';
        select.style.fontSize = '13px';

        var opts = [
            {v: 'en', t: 'English'},
            {v: 'hi', t: 'हिन्दी'},
            {v: 'mr', t: 'मराठी'}
        ];
        opts.forEach(function(o){
            var op = document.createElement('option');
            op.value = o.v;
            op.innerText = o.t;
            select.appendChild(op);
        });

        // Reflect current selection
        var cur = getCookie('googtrans');
        if(cur){
            try{
                var parts = cur.split('/');
                if(parts.length===3){
                    var lang = parts[2];
                    var match = select.querySelector('option[value="'+lang+'"]');
                    if(match) select.value = lang;
                }
            }catch(e){/*ignore*/}
        }

        select.addEventListener('change', function(){
            var lang = this.value || 'en';
            // googtrans format: /<source>/<target>  use auto->target
            setCookie('googtrans','/auto/'+lang,365);
            // Also set a helper cookie so we can reflect selection even before translate loads
            setCookie('__site_lang', lang, 365);
            // Reload to ensure Google Translate picks up the new cookie and translates the page
            window.location.reload();
        });

        var reset = document.createElement('button');
        reset.type = 'button';
        reset.style.border = 'none';
        reset.style.background = '#2e7d32';
        reset.style.color = '#fff';
        reset.style.padding = '6px 10px';
        reset.style.borderRadius = '20px';
        reset.style.cursor = 'pointer';
        reset.style.fontSize = '13px';
        reset.textContent = 'Apply';
        reset.addEventListener('click', function(){
            var lang = select.value || 'en';
            setCookie('googtrans','/auto/'+lang,365);
            setCookie('__site_lang', lang, 365);
            window.location.reload();
        });

        container.appendChild(label);
        container.appendChild(select);
        container.appendChild(reset);

        document.body.appendChild(container);
    }

    // Load Google Translate script with callback
    window.googleTranslateElementInit = function(){
        try{
            new google.translate.TranslateElement({
                pageLanguage: 'en',
                includedLanguages: 'en,hi,mr',
                layout: google.translate.TranslateElement.InlineLayout.SIMPLE,
                autoDisplay: false
            }, 'google_translate_element');
        }catch(e){
            // ignore if script already loaded or fails
            console.warn('googleTranslate init failed', e);
        }
    };

    function loadGoogleScript(){
        if(document.querySelector('script[src*="translate_a/element.js"]')) return;
        var s = document.createElement('script');
        s.src = 'https://translate.google.com/translate_a/element.js?cb=googleTranslateElementInit';
        s.async = true;
        document.head.appendChild(s);
    }

    // On DOM ready
    function ready(fn){
        if(document.readyState!='loading') fn();
        else document.addEventListener('DOMContentLoaded', fn);
    }

    ready(function(){
        // Create a hidden element that Google Translate will use (it can be invisible)
        if(!document.getElementById('google_translate_element')){
            var el = document.createElement('div');
            el.id = 'google_translate_element';
            el.style.display = 'none';
            document.body.appendChild(el);
        }

        createControl();
        // If we have a googtrans cookie already, load Google script so translation is applied
        if(getCookie('googtrans')){
            loadGoogleScript();
        } else {
            // still load script so users can use widget if they want — but keep it light
            loadGoogleScript();
        }
    });
})();
